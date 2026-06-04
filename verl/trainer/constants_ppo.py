# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_SHM_DISABLE": os.environ.get("NCCL_SHM_DISABLE", "0"),
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    # NCCL_SHM_DISABLE must be explicitly passed to Ray workers since they
    # may not inherit the parent env (e.g. small /dev/shm on k8s pods).
    nccl_shm = os.environ.get("NCCL_SHM_DISABLE")
    if nccl_shm:
        runtime_env["env_vars"]["NCCL_SHM_DISABLE"] = nccl_shm

    # Blackwell (SM>=10) + vLLM 0.11 FlashInfer inference-time workaround:
    # force TRT-LLM attention so decode does not fall into the non-TRT-LLM
    # path that asserts on ``decode_wrapper._sm_scale``. Pairs with the
    # cudagraph_mode=PIECEWISE fix in
    # ``verl/workers/rollout/vllm_rollout/vllm_async_server.py::_blackwell_compilation_kwargs``
    # (see that docstring for the root-cause writeup and the removal trigger).
    # ``setdefault`` leaves a user-supplied override alone.
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
            runtime_env["env_vars"].setdefault("VLLM_USE_TRTLLM_ATTENTION", "1")
    except Exception:  # pragma: no cover - defensive: never block ray.init
        pass

    return runtime_env
