### Llama Models

| Model        | Method | Library      | Notebook URL                                                                                                                                                                         | Description                                                                                                                     |
|--------------|--------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Llama 3 8b   | ORPO   | TRL          | [Fine_tune_Llama_3_with_ORPO.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Fine_tune_Llama_3_with_ORPO.ipynb)                                             | Fine-tuning Llama 3 using Odds Ratio Policy Optimization.                                                                       |
| Llama 3 8b   |  -     | LLaMA-Factory| [Finetune_Llama3_with_LLaMA_Factory.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Finetune_Llama3_with_LLaMA_Factory.ipynb)                                | Fine-tuning Llama 3 using the LLaMA-Factory library.                                                                            |
| Llama 3 2.3b (?)   |  -   | Unsloth (?)    | [Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb) | Conversational fine-tuning for Llama 3 (2.1B & 3B) models using faster techniques.  **(Note:  Combined 1B+3B)**                                               |
| Llama ? (?) | SFT (?) | Axolotl (?) |[Fine_tune_LLMs_with_Axolotl.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Fine_tune_LLMs_with_Axolotl.ipynb)| General Llama fine-tuning using Axolotl.  **(Note:  Could be any Llama version.)**|

---

### Gemma Models

| Model             | Method    | Library      | Notebook URL                                                                                                                                                   | Description                                                                                                                       |
|-------------------|-----------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| Gemma 2.0 9b      | SFT       | Axolotl       | [gemma_2_axolotl.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2_axolotl.ipynb)                                            | Fine-tuning Gemma 2 using the Axolotl library.  **(Note:  Assuming this is `gemma_2_axolotl.ipynb`)**                          |
| Gemma 2.0 9b     | QLoRA     | Unsloth      | [gemma_2_9b_qlora_unsloth.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2_9b_qlora_unsloth.ipynb)                                                             | Fine-tuning Gemma 2 9B using QLoRA with Unsloth optimization.                                            |
| Gemma 2.0 9b     | QLoRA     | -            | [gemma_2_9b_qlora.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2_9b_qlora.ipynb)                                                                     | Fine-tuning Gemma 2 9B using QLoRA.                                                                      |
| PaliGemma 2.0 3b  | SFT       | -            | [finetune_paligemma_on_multiple_detection_dataset.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/finetune_paligemma_on_multiple_detection_dataset.ipynb)  | Fine-tuning PaliGemma (Gemma 2 variant) on multiple object detection datasets.                         |
| Gemma 2.0 2b     | Full (?)  | -            | [gemma2(2b)_fc_ft.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma2(2b)_fc_ft.ipynb)                                                                   | Full fine-tuning of the Gemma 2 2B model.                                                                |
| Gemma 2.0 2b       | QLoRA   | -             | [gemma-2_2b_qlora.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma-2_2b_qlora.ipynb)                                                  | Fine-tuning Gemma 2 2B using QLoRA.|
| Gemma 2.0 2b       | QLoRA   | -             | [gemma_2b_qlora.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2b_qlora.ipynb)       | Fine-tuning Gemma 2B using QLoRA.|
| Gemma 1.0 7b     | -         | -            | [Gemma_Fine_tuning.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Gemma_Fine_tuning.ipynb)                                                   | General fine-tuning notebook for Gemma models.                                                            |

---

### Qwen / DeepSeek Models

| Model            | Method        | Library | Notebook URL                                                                                                                                           | Description                                                                                |
|------------------|---------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| DeepSeek 1.0 1.5b| SFT           | TRL     | [Deepseek_Llava_VLM_trl.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Deepseek_Llava_VLM_trl.ipynb)                               | Fine-tuning DeepSeek-Llava VLM using Transformer Reinforcement Learning.                    |
| Qwen 1.0 0.5b    | GRPO          | TRL     | [qwen_grpo_training.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/qwen_grpo_training.ipynb)                                       | GRPO training on the Taylor Swift QA dataset.                                              |

---

### Other Models

| Model           | Method     | Library         | Notebook URL                                                                                                | Description                                                     |
|-----------------|------------|-----------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| GPT-2  (?)      | -          | transformers (?) | [GPT_2_Fine_Tuning_w_Hugging_Face_&_PyTorch.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/GPT_2_Fine_Tuning_w_Hugging_Face_&_PyTorch.ipynb) | GPT-2 Fine-tuning with Hugging Face and PyTorch. |
| Jamba (?)     | -          |  Axolotl       |[LazyAxolotl_Jamba.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/LazyAxolotl_Jamba.ipynb) | Fine tuning using Jamba (Assuming Lazy method).  |
| Phi-3  (?)      |   -         |       -          | [fine_tuning_phi_3_mini_lora.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/fine_tuning_phi_3_mini_lora.ipynb)        | Fine-tuning Phi-3 Mini with LoRA.                           |
| Phi-3  (?)   |   -         | Unsloth (?)        | [fine_tuning_phi_3_mini_lora_unsloth.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/fine_tuning_phi_3_mini_lora_unsloth.ipynb)    | Fine-tuning Phi-3 Mini with LoRA and Unsloth.                |
|  -  | Online DPO | -               | [Online_DPO.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Online_DPO.ipynb)               | Online DPO Implementation.                                     |
| - | -           | - |  [combine_dataset.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/combine_dataset.ipynb)   |  **(Utility:  Dataset Combination)** |
| -       | -           | -  | [torchtune_examples.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/torchtune_examples.ipynb)| TorchTune Examples. |
|GPT 4o (?) |-| - | [openai_gpt_4o_fine_tuning.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/openai_gpt_4o_fine_tuning.ipynb) | Fine Tuning GPT 4o|
---
