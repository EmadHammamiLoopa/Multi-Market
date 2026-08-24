from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor


def _run(cmd: list[str]) -> dict[str, object]:
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"found": False, "command": cmd, "returncode": None, "stdout": "", "stderr": "NOT_FOUND"}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        return {
            "found": True,
            "command": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as exc:
        return {"found": True, "command": cmd, "error": repr(exc)}


def _load_libcuda() -> dict[str, object]:
    candidates = [
        "libcuda.so.1",
        "/usr/lib/wsl/lib/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    ]
    out: list[dict[str, object]] = []
    for name in candidates:
        rec: dict[str, object] = {"candidate": name}
        if name.startswith("/"):
            rec["exists"] = Path(name).exists()
        try:
            ctypes.CDLL(name)
            rec["loadable"] = True
        except Exception as exc:
            rec["loadable"] = False
            rec["error"] = str(exc)
        out.append(rec)
    return {"candidates": out, "any_loadable": any(bool(x.get("loadable")) for x in out)}


def _xgb_probe() -> dict[str, object]:
    x = np.arange(256, dtype=np.float32).reshape(128, 2)
    y = np.sin(np.linspace(0.0, 4.0, 128, dtype=np.float32))
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        max_depth=3,
        learning_rate=0.1,
        n_estimators=8,
        n_jobs=1,
        verbosity=2,
        random_state=20260824,
    )
    caught: list[str] = []
    exc_text = None
    device = None
    pred_ok = False
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            model.fit(x, y)
            pred = model.predict(x[:8])
            pred_ok = bool(len(pred) == 8 and np.all(np.isfinite(pred)))
            caught = [f"{w.category.__name__}: {w.message}" for w in ws]
        cfg = json.loads(model.get_booster().save_config())
        device = cfg.get("learner", {}).get("generic_param", {}).get("device")
    except Exception as exc:
        exc_text = repr(exc)
    return {
        "requested_device": "cuda",
        "resolved_device": device,
        "prediction_ok": pred_ok,
        "warnings": caught,
        "exception": exc_text,
    }


def main() -> int:
    build = xgb.build_info()
    payload = {
        "phase": "V2.3-PHASE0DK-CUDA-DIAGNOSTIC",
        "xgboost_version": xgb.__version__,
        "xgboost_build_info": build,
        "use_cuda_build": build.get("USE_CUDA"),
        "cuda_version_build": build.get("CUDA_VERSION"),
        "nccl_version_build": build.get("NCCL_VERSION"),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "WSL_DISTRO_NAME": os.environ.get("WSL_DISTRO_NAME"),
        },
        "nvidia_smi_L": _run(["nvidia-smi", "-L"]),
        "nvidia_smi_query": _run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]),
        "libcuda": _load_libcuda(),
        "xgboost_probe": _xgb_probe(),
    }
    print(json.dumps(payload, indent=2, default=str))

    ok = bool(
        payload["use_cuda_build"]
        and payload["libcuda"]["any_loadable"]
        and str(payload["xgboost_probe"]["resolved_device"] or "").startswith("cuda")
        and payload["xgboost_probe"]["prediction_ok"]
    )
    print("PHASE0DK_CUDA_DIAGNOSTIC=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
