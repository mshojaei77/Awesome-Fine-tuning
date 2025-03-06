# Awesome-Fine-tuning

A curated list of resources for fine-tuning large language models (LLMs). Contribute to enhance LLM performance and explore various fine-tuning methods with our community!

This repository aims to provide a comprehensive and easily navigable list of fine-tuning resources for popular LLMs. The table below is a starting point, and we encourage contributions to make it even more valuable for the community.

## Fine-tuning Resources Table

| Model Family | Model | Method/Library | Dataset Type | Notebook | Description |
|---|---|---|---|---|---|
| **Gemma** | `Copy_of_qwen_grpo_training.ipynb` | GRPO Training | General | [Copy_of_qwen_grpo_training.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Copy_of_qwen_grpo_training.ipynb) | Notebook for Qwen GRPO (Gradient Ratio Policy Optimization) training adapted for Gemma. |
| **Gemma** | `Gemma_Fine_tuning.ipynb` | General | General | [Gemma_Fine_tuning.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Gemma_Fine_tuning.ipynb) | General fine-tuning notebook for Gemma models. |
| **Gemma** | `finetune_paligemma_on_multiple_detection_dataset.ipynb` | General | Detection Dataset | [finetune_paligemma_on_multiple_detection_dataset.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/finetune_paligemma_on_multiple_detection_dataset.ipynb) | Fine-tuning PaliGemma on multiple object detection datasets. |
| **Gemma** | `gemma-2_2b_qlora.ipynb` | QLoRA | General | [gemma-2_2b_qlora.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma-2_2b_qlora.ipynb) | Fine-tuning Gemma 2 2B model using QLoRA. |
| **Gemma** | `gemma-2_9b_qlora.ipynb` | QLoRA | General | [gemma_2_9b_qlora.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2_9b_qlora.ipynb) | Fine-tuning Gemma 2 9B model using QLoRA. |
| **Gemma** | `gemma_2_9b_qlora_unsloth.ipynb` | QLoRA, Unsloth | General | [gemma_2_9b_qlora_unsloth.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2_9b_qlora_unsloth.ipynb) | Fine-tuning Gemma 2 9B model with QLoRA and Unsloth library for optimization. |
| **Gemma** | `gemma_2_axolotl.ipynb` | Axolotl | General | [gemma_2_axolotl.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2_axolotl.ipynb) | Fine-tuning Gemma 2 model using the Axolotl library. |
| **Gemma** | `gemma_2_axolotl.ipynb` | Axolotl | General | [[Gemma_2]Finetune_with_Axolotl.ipynb]([Gemma_2]Finetune_with_Axolotl.ipynb) | Fine-tuning Gemma 2 model using the Axolotl library. |
| **Gemma** | `gemma2(2b)_fc_ft.ipynb` | Full Fine-tuning | General | [gemma2(2b)_fc_ft.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma2(2b)_fc_ft.ipynb) | Full fine-tuning of Gemma 2 2B model. |
| **Gemma** | `gemma_2b_qlora.ipynb` | QLoRA | General | [gemma_2b_qlora.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/gemma_2b_qlora.ipynb) | Fine-tuning Gemma 2B model using QLoRA. |
| **Gemma** | `[[Gemma_1]Finetune_distributed.ipynb` | Distributed Finetuning | Chat | [[Gemma_1]Finetune_distributed.ipynb]([Gemma_1]Finetune_distributed.ipynb) | Distributed fine-tuning of Gemma model for chat, demonstrating response generation in a pirate's tone. |
| **Gemma** | `[[Gemma_1]Finetune_with_LLaMA_Factory.ipynb` | LLaMA-Factory | General | [[Gemma_1]Finetune_with_LLaMA_Factory.ipynb]([Gemma_1]Finetune_with_LLaMA_Factory.ipynb) | Fine-tuning Gemma model using the LLaMA-Factory library. |
| **Gemma** | `[[Gemma_1]Finetune_with_XTuner.ipynb` | XTuner | General | [[Gemma_1]Finetune_with_XTuner.ipynb]([Gemma_1]Finetune_with_XTuner.ipynb) | Fine-tuning Gemma model using the XTuner library. |
| **Gemma** | `[[Gemma_2]Custom_Vocabulary.ipynb` | Custom Vocabulary | Tokenization | [[Gemma_2]Custom_Vocabulary.ipynb]([Gemma_2]Custom_Vocabulary.ipynb) | Demonstrates using custom vocabulary tokens "<unused[0-98]>" in Gemma models. |
| **Gemma** | `[[Gemma_2]Finetune_with_CALM.ipynb` | CALM | General | [[Gemma_2]Finetune_with_CALM.ipynb]([Gemma_2]Finetune_with_CALM.ipynb) | Fine-tuning Gemma model using the CALM library. |
| **Gemma** | `[[Gemma_2]Finetune_with_Function_Calling.ipynb` | PyTorch/XLA | Function Calling | [[Gemma_2]Finetune_with_Function_Calling.ipynb]([Gemma_2]Finetune_with_Function_Calling.ipynb) | Fine-tuning Gemma for Function Calling using PyTorch/XLA. |
| **Gemma** | `[[Gemma_2]Finetune_with_JORA.ipynb` | JORA | General | [[Gemma_2]Finetune_with_JORA.ipynb]([Gemma_2]Finetune_with_JORA.ipynb) | Fine-tuning Gemma model using the JORA library. |
| **Gemma** | `[[Gemma_2]Finetune_with_LitGPT.ipynb` | LitGPT | General | [[Gemma_2]Finetune_with_LitGPT.ipynb]([Gemma_2]Finetune_with_LitGPT.ipynb) | Fine-tuning Gemma model using the LitGPT library. |
| **Gemma** | `[[Gemma_2]Finetune_with_Torch_XLA.ipynb` | PyTorch/XLA | General | [[Gemma_2]Finetune_with_Torch_XLA.ipynb]([Gemma_2]Finetune_with_Torch_XLA.ipynb) | Fine-tuning Gemma model using PyTorch/XLA. |
| **Gemma** | `[[Gemma_2]Finetune_with_Unsloth.ipynb` | Unsloth | General | [[Gemma_2]Finetune_with_Unsloth.ipynb]([Gemma_2]Finetune_with_Unsloth.ipynb) | Fine-tuning Gemma model using the Unsloth library. |
| **Gemma** | `[[Gemma_2]Translator_of_Old_Korean_Literature.ipynb` | Keras | Translation | [[Gemma_2]Translator_of_Old_Korean_Literature.ipynb]([Gemma_2]Translator_of_Old_Korean_Literature.ipynb) | Using Gemma model to translate old Korean literature with Keras. |
| **GPT** | `GPT_2_Fine_Tuning_w_Hugging_Face_&_PyTorch.ipynb` | Hugging Face, PyTorch | General | [GPT_2_Fine_Tuning_w_Hugging_Face_&_PyTorch.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/GPT_2_Fine_Tuning_w_Hugging_Face_&_PyTorch.ipynb) | Fine-tuning GPT-2 model with Hugging Face Transformers and PyTorch. |
| **GPT** | `openai_gpt_4o_fine_tuning.ipynb` | General | General | [openai_gpt_4o_fine_tuning.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/openai_gpt_4o_fine_tuning.ipynb) | Fine-tuning notebook for OpenAI's GPT-4o model. |
| **Llama** | `Deepseek_Llava_VLM_trl.ipynb` | TRL | VLM | [Deepseek_Llava_VLM_trl.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Deepseek_Llava_VLM_trl.ipynb) | Fine-tuning Deepseek-Llava VLM using Transformer Reinforcement Learning (TRL). |
| **Llama** | `Fine_tune_Llama_3_with_ORPO.ipynb` | ORPO | General | [Fine_tune_Llama_3_with_ORPO.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Fine_tune_Llama_3_with_ORPO.ipynb) | Fine-tuning Llama 3 model with Odds Ratio Policy Optimization (ORPO). |
| **Llama** | `Finetune_Llama3_with_LLaMA_Factory.ipynb` | LLaMA-Factory | General | [Finetune_Llama3_with_LLaMA_Factory.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Finetune_Llama3_with_LLaMA_Factory.ipynb) | Fine-tuning Llama 3 model using LLaMA-Factory. |
| **Llama** | `Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb` | General, Faster Finetuning | Conversational | [Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning.ipynb) | Conversational fine-tuning for Llama 3 2.1B and 3B models with faster fine-tuning techniques. |
| **Other** | `LazyAxolotl_Jamba.ipynb` | Axolotl | General | [LazyAxolotl_Jamba.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/LazyAxolotl_Jamba.ipynb) | Fine-tuning Jamba model using LazyAxolotl. |
| **Other** | `torchtune_examples.ipynb` | TorchTune | Examples | [torchtune_examples.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/torchtune_examples.ipynb) | Example notebooks demonstrating fine-tuning with TorchTune. |
| **Other** | `⚡Online_DPO.ipynb` | Online DPO | General | [⚡Online_DPO.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/%E2%9A%A1Online_DPO.ipynb) | Notebook for Online Direct Preference Optimization (DPO). |
| **Phi** | `fine_tuning_phi_3_mini_lora.ipynb` | LoRA | General | [fine_tuning_phi_3_mini_lora.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/fine_tuning_phi_3_mini_lora.ipynb) | Fine-tuning Phi-3-mini model using LoRA. |
| **Phi** | `fine_tuning_phi_3_mini_lora_unsloth.ipynb` | LoRA, Unsloth | General | [fine_tuning_phi_3_mini_lora_unsloth.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/fine_tuning_phi_3_mini_lora_unsloth.ipynb) | Fine-tuning Phi-3-mini model with LoRA and Unsloth library. |
| **Yi** | `Fine_tune_LLMs_with_Axolotl.ipynb` | Axolotl | General | [Fine_tune_LLMs_with_Axolotl.ipynb](https://github.com/mshojaei77/Awesome-Fine-tuning/blob/main/Fine_tune_LLMs_with_Axolotl.ipynb) | General notebook for fine-tuning LLMs using Axolotl library, demonstrated with Yi models. |

## Contribution Guide

We highly encourage contributions to expand this resource list! If you have a notebook or resource related to fine-tuning any LLM, please contribute by following these steps:

1.  **Fork this repository.**
2.  **Add a new row to the `Fine-tuning Resources Table`** in `readme.md` with the following information:
    *   **Model Family:**  (e.g., Gemma, Llama, Mistral) - Group models from the same family together.
    *   **Model:** The specific model name (e.g., `gemma-2b`, `Llama-3-8B`). Ideally, use the exact model name from Hugging Face Hub or the original source.
    *   **Method/Library:** The fine-tuning method or library used (e.g., LoRA, QLoRA, Full Fine-tuning, Axolotl, LLaMA-Factory, TRL).
    *   **Dataset Type:**  The type of dataset used for fine-tuning (e.g., Chat, Translation, Summarization, Code Generation, Instruction Following, Detection Dataset). Use "General" if it's a broadly applicable notebook.
    *   **Notebook:** A direct link to your notebook. Preferably host notebooks in a public repository like GitHub or Google Colab.
    *   **Description:** A concise and informative description of the notebook, highlighting what it demonstrates or the task it achieves.

3.  **Submit a Pull Request** with your changes.

By contributing, you will help make this repository an even more valuable and comprehensive resource for the LLM community! Let's collaborate and build the best Awesome-Fine-tuning list together!
