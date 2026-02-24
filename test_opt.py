# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
os.environ.setdefault("VLLM_USE_NONSTRICT_COMPILE", "1")

from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        model="meta-llama/Meta-Llama-3-70B",
        tensor_parallel_size=4,
        # compilation_config={"cudagraph_mode": "none", "backend": "eager"},
        compilation_config={
            "cudagraph_mode": "none",
            "pass_config": {"fuse_allreduce_rms": False},
        },
    )
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

    prompts = ["The capital of France is"]
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
