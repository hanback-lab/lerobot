# pusht_eval.py
import os
import time
import numpy as np
import torch
import gymnasium as gym

import gym_pusht
from huggingface_hub import snapshot_download
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def make_env(obs_type="state", render=False, seed=0):
    """
    PushT 환경을 생성합니다.
    obs_type: "state" 또는 "pixels"
    render  : 렌더링이 필요하면 True (Jetson에선 보통 False 권장)
    """
    render_mode = "rgb_array" if render else None
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
    try:
        env.reset(seed=seed)
    except TypeError:
        # gym 버전에 따라 reset(seed=...) 미지원일 수 있음
        pass
    return env


def load_policy(repo_or_path, device=None):
    """
    HF Hub 레포 혹은 로컬 경로에서 사전학습 정책을 로드합니다.
    """
    if os.path.isdir(repo_or_path):
        local_dir = repo_or_path
    else:
        # HF에서 로컬로 내려받기
        local_dir = snapshot_download(repo_id=repo_or_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = DiffusionPolicy.from_pretrained(local_dir)
    policy.eval()
    return policy

def to_policy_input(obs, obs_type="state", device="cpu"):
    import torch, numpy as np

    if obs_type == "state":
        # (이전과 동일) obs가 dict/ndarray 모두 대응
        arr = obs["state"] if isinstance(obs, dict) and "state" in obs else obs
        arr = np.asarray(arr, dtype=np.float32)
        x = torch.from_numpy(arr).to(device).unsqueeze(0)  # [1, D]
        return {"observation.state": x}

    elif obs_type == "pixels":
        # obs 가 보통 HxWx3 uint8 (예: 96x96x3)
        if isinstance(obs, dict):
            # 혹시 dict 형태로 올 경우 키 이름에 따라 수정
            img = obs.get("pixels", None) or obs.get("image", None) or next(iter(obs.values()))
        else:
            img = obs  # ndarray (H,W,3)

        img = np.asarray(img)
        # [H,W,3] → [1,3,H,W], float32 [0,1]
        img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
        return {"observation.image": img_t}

def rollout(env, policy, n_episodes=10, obs_type="state", action_clip=(0.0, 512.0)):
    """
    평가 루프:
    - 각 에피소드에서 policy가 출력한 k-step 액션 청크를 순차 실행
    - 보상(coverage)의 최대값을 Max Overlap으로 기록
    - Max Overlap ≥ 0.95면 성공으로 집계
    """
    device = next(policy.parameters()).device if hasattr(policy, "parameters") else "cpu"

    succ = 0
    max_overlaps = []
    step_times = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_max = 0.0

        while not done:
            # 정책 입력 만들기
            inp = to_policy_input(obs, obs_type="pixels", device=device)

            with torch.inference_mode():
                t0 = time.time()
                # ACT/Diffusion 등 사전학습 정책은 보통 "청크(k-step) 액션 시퀀스"를 반환
                chunk = policy(inp)
                step_times.append((time.time() - t0) * 1000.0)

            # numpy로 변환
            if isinstance(chunk, (tuple, list)):
                act_seq = np.asarray(chunk[0])  # 정책 구현에 따라 첫 요소에 액션이 있을 수 있음
            else:
                act_seq = np.asarray(chunk)

            # 청크 순차 실행
            for a in act_seq:
                # PushT 연속 동작: [x, y] in [0,512]
                a = np.asarray(a, dtype=np.float32)
                if action_clip is not None:
                    lo, hi = action_clip
                    a = np.clip(a, lo, hi)

                obs, reward, terminated, truncated, info = env.step(a)
                ep_max = max(ep_max, float(reward))
                if terminated or truncated:
                    done = True
                    break

        max_overlaps.append(ep_max)
        succ += (ep_max >= 0.95)

    return {
        "avg_max_overlap": float(np.mean(max_overlaps)) if max_overlaps else 0.0,
        "success_rate": float(succ) / float(n_episodes) if n_episodes > 0 else 0.0,
        "avg_infer_ms": float(np.mean(step_times)) if step_times else 0.0,
        "episodes": n_episodes,
    }


def main():
    # ✅ 1) 테스트용으로 안정적인 공식 모델(확인용)
    #    파이프라인 체크 목적: diffusion 정책 (팀 공식 업로드가 흔히 더 호환이 좋음)
    OFFICIAL_MODEL = "lerobot/diffusion_pusht"

    # ✅ 2) ACT 모델로 바로 테스트하고 싶다면 아래로 바꾸세요.
    #    HF 업로더/버전에 따라 config 불일치가 있을 수 있으니 먼저 OFFICIAL_MODEL로 동작 확인 권장.
    # ACT_MODEL = "aadarshram/act_pusht"

    model_repo = OFFICIAL_MODEL  # 또는 ACT_MODEL

    # 환경/정책 준비
    env = make_env(obs_type="state", render=False, seed=0)
    policy = load_policy(model_repo, device=None)

    # 평가
    stats = rollout(env, policy, n_episodes=10, obs_type="state")
    env.close()

    print("=== PushT Eval Results ===")
    print(f"Episodes        : {stats['episodes']}")
    print(f"Avg Max Overlap : {stats['avg_max_overlap']:.4f}")
    print(f"Success Rate    : {stats['success_rate']:.3f}")
    print(f"Avg Infer (ms)  : {stats['avg_infer_ms']:.2f}")


if __name__ == "__main__":
    main()
