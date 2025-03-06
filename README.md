### Llama Models

| Model        | Method | Library      | Notebook URL                                                                                                                                                                         | Description                                                                                                                     |
|--------------|--------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Llama 3.2 3b   | QLoRA  | Unsloth | [Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb) | Conversational fine-tuning for Llama 3 (2.1B & 3B) models using faster techniques.                                               |
| Llama 3 8b   | ORPO   | TRL | [Fine_tune_Llama_3_with_ORPO.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Fine_tune_Llama_3_with_ORPO.ipynb)                                             | Fine-tuning Llama 3 using Odds Ratio Policy Optimization.                                                                       |
| Llama 3 8b   | QLoRA | LLaMA-Factory | [Finetune_Llama3_with_LLaMA_Factory.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Finetune_Llama3_with_LLaMA_Factory.ipynb)                                | Fine-tuning Llama 3 using the LLaMA-Factory library.                                                                            |

---

### Gemma Models

| Model             | Method | Library      | Notebook URL                                                                                                                                                   | Description                                                                                           |
|-------------------|--------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **Gemma 2**        |        |              |                                                                                     | **Gemma 2 Models**                                                                            |
| Gemma 2.0  9b     | SFT | Axolotl       | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_2%5DFinetune_with_Axolotl.ipynb)                                                  | Fine-tuning Gemma 2 using the Axolotl library.                                                         |
| Gemma 2.0  9b   |  SFT |    -          | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_2%5DFinetune_with_CALM.ipynb)                                                     | Fine-tuning Gemma 2 using the CALM library.                                                              |
| Gemma 2.0 9b     | SFT |    -          | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_2%5DFinetune_with_Function_Calling.ipynb)                                           | Fine-tuning Gemma 2 for function calling with PyTorch/XLA.                                             |
| Gemma 2.0 9b     | SFT |     -         | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_2%5DFinetune_with_JORA.ipynb)                                                     | Fine-tuning Gemma 2 using the JORA library.                                                              |
| Gemma 2.0 9b      | SFT |      -        | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_2%5DFinetune_with_LitGPT.ipynb)                                                   | Fine-tuning Gemma 2 using the LitGPT library.                                                          |
| Gemma 2.0 9b     | SFT |      -        | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_2%5DFinetune_with_Torch_XLA.ipynb)                                                | Fine-tuning Gemma 2 using PyTorch/XLA.                                                                 |
| Gemma 2.0 9b       | SFT | Unsloth      | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_2%5DFinetune_with_Unsloth.ipynb)                                                  | Fine-tuning Gemma 2 using the Unsloth library.                                                         |
| Gemma 2.0 9b   | SFT |      -        | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_2%5DTranslator_of_Old_Korean_Literature.ipynb)                                       | Using Gemma 2 to translate old Korean literature with Keras.                                         |
| PaliGemma 2.0 3b | SFT |        -      | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/finetune_paligemma_on_multiple_detection_dataset.ipynb)                                      | Fine-tuning PaliGemma (Gemma 2 variant) on multiple object detection datasets.                         |
| Gemma 2.0 2b     | QLoRA  |       -       | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma-2_2b_qlora.ipynb)                                                                     | Fine-tuning Gemma 2 2B using QLoRA.                                                                      |
| Gemma 2.0 9b     | QLoRA  |      -        | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2_9b_qlora.ipynb)                                                                     | Fine-tuning Gemma 2 9B using QLoRA.                                                                      |
| Gemma 2.0 9b     | QLoRA  | Unsloth      | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2_9b_qlora_unsloth.ipynb)                                                             | Fine-tuning Gemma 2 9B using QLoRA with Unsloth optimization.                                            |
| Gemma 2.0 2b     |   -   |       -       | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma2(2b)_fc_ft.ipynb)                                                                     | Full fine-tuning of the Gemma 2 2B model.                                                                |
| Gemma 2.0 2b       | QLoRA  |      -      | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2b_qlora.ipynb)                                                                       | Fine-tuning Gemma 2B using QLoRA.                                                                        |
| **Gemma 1**    |        |              |  |                                                                                     | **Gemma 1 Models**                                                                            |
| Gemma 1.0 7b     |     -   | LLaMA-Factory | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_1%5DFinetune_with_LLaMA_Factory.ipynb)                                              | Fine-tuning Gemma (version 1) using LLaMA-Factory.                                                     |
| Gemma 1.0 7b   |  -  |    -          | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_1%5DFinetune_with_XTuner.ipynb)                                                     | Fine-tuning Gemma (version 1) using the XTuner library.                                                |
| Gemma 1.0 7b   |    -    |       -       | [Link](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%5BGemma_1%5DFinetune_distributed.ipynb)                                                     | Distributed fine-tuning for chat applications with Gemma (version 1).                                    |
| **Gemma**         |        |              |                                                                                       | **Original Gemma Model**                                                                            |
| Gemma 1.0 7b     |    -    |       -       | [Gemma_Fine_tuning.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Gemma_Fine_tuning.ipynb)                                                   | General fine-tuning notebook for Gemma models.                                                            |

---

### Qwen / DeepSeek Models

| Model            | Method        | Library | Notebook URL                                                                                                                                           | Description                                                                                |
|------------------|---------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| DeepSeek 1.0 1.5b | SFT           | TRL     | [Deepseek_Llava_VLM_trl.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Deepseek_Llava_VLM_trl.ipynb)                               | Fine-tuning DeepSeek-Llava VLM using Transformer Reinforcement Learning.                    |
| Qwen 1.0 0.5b    | GRPO Training | TRL     | [qwen_grpo_training.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/qwen_grpo_training.ipynb)                                       | GRPO training on the Taylor Swift QA dataset.                                              |

---
