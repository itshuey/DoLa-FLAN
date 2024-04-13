Enhancing Instruction-Following Capabilities in Seq2Seq Models: A Novel Adaptation of DoLa in T5 and FLAN-T5
===
Work by Anabel Y., Lorenzo G., Huey S., Felipe J, Anthony H., Harman C.

Based on a fork of "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models"

Paper: https://arxiv.org/abs/2309.03883  
Authors: [Yung-Sung Chuang](https://people.csail.mit.edu/yungsung/) $^\dagger$, [Yujia Xie](https://sites.google.com/view/yujia) $^\ddagger$, [Hongyin Luo](https://luohongyin.github.io/) $^\dagger$, [Yoon Kim](https://people.csail.mit.edu/yoonkim/) $^\dagger$, [James Glass](https://people.csail.mit.edu/jrg/) $^\dagger$, [Pengcheng He](https://scholar.google.com/citations?user=TS1RoxAAAAAJ&hl=en) $^\ddagger$  
$^\dagger$ Massachusetts Institute of Technology, $^\ddagger$ Microsoft

# Overview
The repository for "Enhancing Instruction-Following Capabilities in Seq2Seq Models: A Novel Adaptation of DoLa in T5 and FLAN-T5".

## DoLA strategy

A novel decoding strategy aimed at reducing hallucinations in large language models (LLMs) without the need for external knowledge retrieval or additional fine-tuning is adapted to improve on instruction-following in seq2seq. 

At its core, DoLA (Decoding by contrasting layers) contrasts the knowledge embedded in different layers of a transformer model. By comparing the logits from earlier (less mature) and later (more mature) layers, DoLA aims to filter out hallucinated content, favoring more factual outputs.

## Main Files (with Detailed Docs)
- `dola_t5.py` A class DoLaT5 is defined for working with a T5 model, supporting various generation and scoring methods, including baseline, DOLA-static, and DOLA modes. It's designed to run on either CPU or GPU, with support for multi-GPU setups.
- `ifeval_eval.py` Script to evaluate the language model's performance on a given dataset. Uses the Hugging Face Transformers library to load and interact with pre-trained models. It handles different configurations and modes of operation, including parallel processing and early exit strategies for efficient inference.
- `memotrap_dataset_eval.py` Script to evaluate the performance of language models, specifically focusing on their ability to generate correct endings for given prompts. It utilizes a dataset loaded from a CSV file and supports different configurations and modes for the language model, including the use of DoLa and DoLaT5 models for improved factuality.

## Repository Structure and Key Components
- `Scripts and Usage:` The provided scripts are straightforward to use, requiring only the specification of the model, dataset paths, and the desired decoding strategy through command-line arguments. This design makes it easy to replicate the experiments or apply DoLA to new models and tasks.

- `Evaluation Framework:` The inclusion of evaluation scripts for specific tasks and datasets, along with instructions for using external tools for response comparison, offers a comprehensive framework for assessing the effectiveness of DoLA in enhancing the factuality of LLMs.

# `dola_t5.py` Docs
The `dola_t5.py` script defines a class DoLaT5 for working with a T5 model, supporting various generation and scoring methods, including baseline, DOLA-static, and DOLA modes. It's designed to run on either CPU or GPU, with support for multi-GPU setups. The script is structured into several key components:

1. Initialization: The __init__ method initializes the class with model details, device configuration, and loads the model and tokenizer.

2. Model Loading: `load_model` loads the T5 model and tokenizer. It configures the model for efficient memory usage on GPUs and supports distributing the model across multiple GPUs if specified.

3. Stopping Criteria: `set_stop_words` allows setting custom stopping criteria for generation, using the T5StoppingCriteria.

4. Text Generation: The `generate` method supports text generation in three modes:

- `baseline`: Standard text generation.

- `dola-static`: Uses specified mature and premature layers for DOLA decoding.

- `dola`: Dynamically selects the premature layer based on divergence from the mature layer's output.

It supports various generation parameters like `max_new_tokens, top_p, top_k`, and temperature. The method can also remove specified stop words from the output.

5. Relative Top Filtering: `get_relative_top_filter` is a utility for applying a relative top filter based on the scores' softmax values, used in DOLA modes for filtering logits.

6. Language Modeling Score: `lm_score` calculates the language modeling score for a given text, supporting the same three modes as text generation. It can compute scores based on the difference in logits between layers (for DOLA modes) and supports PMI calculation.

7. Utility Methods: The script includes methods for softmax normalization, KL divergence calculation, and JS divergence calculation for selecting the premature layer in DOLA mode.

Key functionalities include:

- DOLA Decoding: Dynamically selects layers for decoding based on divergence, aiming to improve generation quality.

- Efficient Memory Usage: Configures the model for low memory usage on CPUs and efficient distribution across multiple GPUs.

- Custom Stopping Criteria: Allows specifying custom stopping words for generation tasks.


# `ifeval_eval.py` Docs
Script to evaluate the language model's performance on a given dataset. Uses the Hugging Face Transformers library to load and interact with pre-trained models. It handles different configurations and modes of operation, including parallel processing and early exit strategies for efficient inference.

It is structured as follows:
1. Imports and Setup: The script imports necessary libraries and sets up regular expressions and constants. It suppresses logging messages from the Transformers library to reduce clutter.

2. Functions:

- load_jsonl(file_path): Loads a JSONL file and returns a list of prompts extracted from it.

- create_demo_text(): Creates a demonstration text with questions and answers to be prepended to the input prompts.

- build_prompt(input_text): Builds the final prompt by appending the input text to the demonstration text.

3. Argument Parsing: The script uses argparse to parse command-line arguments, allowing users to specify the model name, device, data path, and other configurations.

4. Data Preparation: It loads the dataset from a specified path and optionally limits the number of prompts for debugging or splits the dataset for parallel processing.

5. Model Initialization: Depending on the model name, it initializes either DoLa or DoLaT5 class, which are presumably custom classes for handling language model inference. The script sets stop words to signal the end of a generation.

6. Early Exit Layers Configuration: It configures early exit layers for the model, which is a technique to improve inference efficiency by exiting the model's forward pass early under certain conditions. The script supports three modes:

- baseline: Standard decoding without early exit.

- early_exit_contrastive: Uses a specific mature and premature layer for early exit.

- dola: Dynamically chooses from a set of candidate premature layers based on certain criteria.

7. Inference Loop: For each prompt in the dataset, the script:

- Builds the full prompt using build_prompt.

- Generates a completion using the model with specified generation parameters.

- Cleans up the generated text by removing stop words.

- Optionally, tracks the usage of premature layers in dola mode.

- Results Handling: The script collects the prompts and their corresponding model completions in a list of dictionaries.

9. Output: Finally, it saves the results to a JSONL file in the specified output path. If parallel processing is enabled, it appends the shard ID to the output filename.

# `memotrap_dataset_eval.py` Docs
Script to evaluate the performance of language models, specifically focusing on their ability to generate correct endings for given prompts. It utilizes a dataset loaded from a CSV file and supports different configurations and modes for the language model, including the use of DoLa and DoLaT5 models for improved factuality.
1. Imports and Initial Setup: The script imports necessary libraries and sets up logging and constants. It defines regular expressions for parsing answers and initializes flags for debugging and other configurations.

2. Utility Functions:

- parse_classes: Parses a string representation of a list into an actual list of strings.

- load_csv: Loads data from a CSV file, parsing each line into a dictionary with keys for the prompt, possible classes (answers), and the correct answer index.

- extract_and_compare_answer: Extracts the model's generated answer ending and compares it with the correct answer to determine correctness.

- create_demo_text: Generates a demo text with example questions and answers to be used in the prompt construction.

- build_prompt: Constructs the input prompt for the model by appending the demo text and the specific question to be answered.

3. Argument Parsing: The script uses argparse to handle command-line arguments for model configuration, dataset paths, and evaluation settings.

4. Model Selection and Configuration: Based on the provided model name, the script selects between the DoLa and DoLaT5 models. It also sets up model-specific configurations like stop words, early exit layers, and repetition penalties.

5. Data Preparation: The script loads the dataset from a CSV file. It supports debugging mode (which limits the data to the first 10 samples) and parallel processing mode (which divides the dataset into chunks based on shard IDs).

6. Evaluation Loop:

- For each sample in the dataset, it constructs the input prompt and generates model completions based on the provided arguments.

- It then cleans the model completion by removing any stop words and trims whitespace.

- The script extracts the model's answer ending and compares it with the correct answer to determine correctness.

- It accumulates results, including the model's completions, the generated answer endings, the correct answers, and correctness flags.

7. Results Reporting and Saving:

- Calculates the overall accuracy of the model based on the correctness flags.

- In "dola" mode with debugging enabled, it reports the usage statistics of premature layers.

- Saves the evaluation results to a JSON file, with the filename optionally including the shard ID for parallel processing setups.