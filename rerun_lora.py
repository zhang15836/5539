from modernbert_sst2 import run_lora
import json, os

if __name__ == "__main__":
    lora_result = run_lora(592130)  # 592130 是之前 head-tuning 打印出的 trainable params
    print(lora_result)
    os.makedirs("runs", exist_ok=True)
    with open("runs/lora_r1_result.json", "w") as f:
        json.dump(lora_result, f, indent=2)

