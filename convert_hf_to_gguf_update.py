#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import pathlib
import re

import requests
import json
import shutil
import argparse

from hashlib import sha256
from enum import IntEnum, auto
from transformers import AutoTokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("convert_hf_to_gguf_update")
sess = requests.Session()

convert_py_pth = pathlib.Path("conversion/base.py")
convert_py = convert_py_pth.read_text(encoding="utf-8")
hf_token_pth = pathlib.Path.home() / ".cache" / "huggingface" / "token"
hf_token = hf_token_pth.read_text(encoding="utf-8").strip() if hf_token_pth.exists() else None


class TOKENIZER_TYPE(IntEnum):
    SPM = auto()
    BPE = auto()
    WPM = auto()
    UGM = auto()


DOC_STRING = """
This script downloads the tokenizer models of the specified models from Huggingface and
generates the get_vocab_base_pre() function for convert_hf_to_gguf.py

/!\\ It is intended to be used by contributors and is not meant to be run by end users

This is necessary in order to analyze the type of pre-tokenizer used by the model and
provide the necessary information to llama.cpp via the GGUF header in order to implement
the same pre-tokenizer.

ref: https://github.com/ggml-org/llama.cpp/pull/6920

Instructions:

- Add a new model to the "models" list
- Run the script with your huggingface token
    By default, token will be read from ~/.cache/huggingface/token
- The convert_hf_to_gguf.py script will have had its get_vocab_base_pre() function updated
- Update llama.cpp with the new pre-tokenizer if necessary
"""
# TODO: generate tokenizer tests for llama.cpp

parser = argparse.ArgumentParser(description=DOC_STRING, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--full", action="store_true",
    help="download full list of models - make sure you have access to all of them",
)
parser.add_argument(
    "--check-missing", action="store_true",
    help="only check for missing pre-tokenizer hashes",
)
parser.add_argument(
    "hf_token",
    help="optional HF token",
    nargs="?",
)
args = parser.parse_args()
hf_token = args.hf_token if args.hf_token is not None else hf_token

if hf_token is None:
    logger.warning("HF token not found. You can provide it as an argument or set it in ~/.cache/huggingface/token")

if args.check_missing and args.full:
    logger.warning("Downloading full list of models requested, ignoring --check-missing!")
    args.check_missing = False

# TODO: this string has to exercise as much pre-tokenizer functionality as possible
#       will be updated with time - contributions welcome
CHK_TXT = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````\"\"\"\"......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

# TODO: add models here, base models preferred
models = [
    {"name": "inkling2",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/thinkingmachines/Inkling-2-Small", },
    {"name": "bailingmoe2",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/inclusionAI/Ling-mini-base-2.0", },
    {"name": "qwen35",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/Qwen/Qwen3.5-9B-Instruct", },
]

# some models are known to be broken upstream, so we will skip them as exceptions
pre_computed_hashes: list[dict] = [
    {"name": "qwen35",    "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/openbmb/MiniCPM-V-4_6", "chkhsh": "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f"},
    # Kolibri-1 and Kolibri-1.1 use a 128k vocabulary with the qwen2 pre-tokenizer regex
    {"name": "qwen2",     "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/Aleph-Alpha/Kolibri-1.1", "chkhsh": "6e040dfe72e4b85855588c53acf4909ac4e98e55bc3a33cd7db499180dc42a78"},
]


def download_file_with_auth(url, token, save_path):
    headers = {"Authorization": f"Bearer {token}"} if token else None
    response = sess.get(url, headers=headers)
    response.raise_for_status()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as downloaded_file:
        downloaded_file.write(response.content)
    logger.info(f"File {save_path} downloaded successfully")


def download_model(model):
    name = model["name"]
    repo = model["repo"]
    tokt = model["tokt"]

    files = ["config.json", "tokenizer.json", "tokenizer_config.json"]

    if tokt == TOKENIZER_TYPE.SPM:
        files.append("tokenizer.model")

    if tokt == TOKENIZER_TYPE.UGM:
        files.append("spiece.model")

    if os.path.isdir(repo):
        # If repo is a path on the file system, copy the directory
        os.makedirs(f"models/tokenizers/{name}", exist_ok=True)
        for file in files:
            src_path = os.path.join(repo, file)
            dst_path = f"models/tokenizers/{name}/{file}"
            if os.path.isfile(dst_path):
                logger.info(f"{name}: File {dst_path} already exists - skipping")
                continue
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"{name}: Copied {src_path} to {dst_path}")
            else:
                logger.warning(f"{name}: Source file {src_path} does not exist")
    else:
        # If repo is a URL, download the files
        for file in files:
            save_path = f"models/tokenizers/{name}/{file}"
            if os.path.isfile(save_path):
                logger.info(f"{name}: File {save_path} already exists - skipping")
                continue
            download_file_with_auth(f"{repo}/resolve/main/{file}", hf_token, save_path)


# get list of existing models and chkhsh from the convert_hf_to_gguf.py file
# returns mapping res --> chkhsh
def get_existing_models(convert_py):
    pattern = r'if chkhsh == "([a-f0-9]{64})":\s*\n\s*.*\s*res = "([^"]+)"'
    matches = re.findall(pattern, convert_py)
    output = {}
    for chkhsh, res in matches:
        output[res] = chkhsh
    return output


existing_models = {}
all_models = models.copy()
if not args.full:
    # Filter out models that already exist in convert_hf_to_gguf.py
    existing_models = get_existing_models(convert_py)
    all_models = models.copy()
    models = [model for model in all_models if model["name"] not in existing_models]

if not args.check_missing:
    logging.info(f"Downloading {len(models)} models...")
    for model in models:
        try:
            download_model(model)
        except Exception as e:
            logger.error(f"Failed to download model {model['name']}. Error: {e}")


# generate the source code for the convert_hf_to_gguf.py:get_vocab_base_pre() function:

src_ifs = ""
for model in [*pre_computed_hashes, *all_models]:
    name = model["name"]
    tokt = model["tokt"]
    chkhsh = model.get("chkhsh")

    if tokt == TOKENIZER_TYPE.SPM or tokt == TOKENIZER_TYPE.UGM:
        continue

    # create the tokenizer
    if chkhsh is not None:
        # if the model has a pre-computed hash, use it
        logger.info(f"Using pre-computed hash for model {name}: {chkhsh}")
    elif name in existing_models:
        # if the model already exists in convert_hf_to_gguf.py, skip compute hash
        chkhsh = existing_models[name]
    else:
        # otherwise, compute the hash of the tokenizer

        # Fail if the tokenizer folder with config does not exist or there are other download issues previously
        if not os.path.isfile(f"models/tokenizers/{name}/tokenizer_config.json"):
            raise OSError(f"Config for tokenizer {name} not found. The model may not exist or is not accessible with the provided token.")

        try:
            logger.info(f"Loading tokenizer from {f'models/tokenizers/{name}'}...")
            tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}")
        except Exception as e:
            raise OSError(f"Error loading tokenizer for model {name}.") from e

        assert tokenizer is not None
        chktok = tokenizer.encode(CHK_TXT)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.info(f"model: {name}")
        logger.info(f"tokt: {tokt}")
        logger.info(f"repo: {model['repo']}")
        logger.info(f"chktok: {chktok}")
        logger.info(f"chkhsh: {chkhsh}")

        # print the "pre_tokenizer" content from the tokenizer.json
        with open(f"models/tokenizers/{name}/tokenizer.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            normalizer = cfg["normalizer"]
            logger.info("normalizer: " + json.dumps(normalizer, indent=4))
            pre_tokenizer = cfg["pre_tokenizer"]
            logger.info("pre_tokenizer: " + json.dumps(pre_tokenizer, indent=4))
            if "ignore_merges" in cfg["model"]:
                logger.info("ignore_merges: " + json.dumps(cfg["model"]["ignore_merges"], indent=4))

        logger.info("")

    src_ifs += f"        if chkhsh == \"{chkhsh}\":\n"
    src_ifs += f"            # ref: {model['repo']}\n"
    src_ifs += f"            res = \"{name}\"\n"

src_func = f"""
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = {repr(CHK_TXT)}

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {{chktok}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
{src_ifs}
        if res is None:
            logger.warning("\\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert_hf_to_gguf_update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggml-org/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {{chkhsh}}")
            logger.warning("**************************************************************************************")
            logger.warning("\\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {{repr(res)}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        return res
"""

convert_py = re.sub(
    r"(# Marker: Start get_vocab_base_pre)(.+?)( +# Marker: End get_vocab_base_pre)",
    lambda m: m.group(1) + src_func + m.group(3),
    convert_py,
    flags=re.DOTALL | re.MULTILINE,
)

convert_py_pth.write_text(convert_py, encoding="utf-8")

logger.info(f"+++ {convert_py_pth} was updated")

# generate tests for each tokenizer model

tests = [
    "ied 4 ½ months",
    "Äpfel",
    "",
    " ",
    "  ",
    "   ",
    "\t",
    "\n",
    "\n\n",
    "\n\n\n",
    "\t\n",
    "Hello world",
    " Hello world",
    "Hello World",
    " Hello World",
    " Hello World!",
    "Hello, world!",
    " Hello, world!",
    " this is 🦙.cpp",
    "w048 7tuijk dsdfhu",
    "нещо на Български",
    "កាន់តែពិសេសអាចខលចេញ",
    "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)",
    "Hello",
    " Hello",
    "  Hello",
    "   Hello",
    "    Hello",
    "    Hello\n    Hello",
    " (",
    "\n =",
    "' era",
    "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
    "!!!!!!",
    "3",
    "33",
    "333",
    "3333",
    "33333",
    "333333",
    "3333333",
    "33333333",
    "333333333",
    "Cửa Việt", # llama-bpe fails on this
    " discards",
    CHK_TXT,
]

# write the tests to ./models/ggml-vocab-{name}.gguf.inp
# the format is:
#
# test0
# __ggml_vocab_test__
# test1
# __ggml_vocab_test__
# ...
#

# with each model, encode all tests and write the results in ./models/ggml-vocab-{name}.gguf.out
# for each test, write the resulting tokens on a separate line

for model in models:
    name = model["name"]
    tokt = model["tokt"]

    # Skip if the tokenizer folder does not exist or there are other download issues previously
    if not os.path.exists(f"models/tokenizers/{name}"):
        logger.warning(f"Directory for tokenizer {name} not found. Skipping...")
        continue

    # create the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}")
    except (OSError, TypeError, ValueError) as e:
        logger.error(f"Failed to load tokenizer for model {name}. Error: {e}")
        continue  # Skip this model and continue with the next one in the loop

    assert tokenizer is not None
    if not os.path.exists(f"models/ggml-vocab-{name}.gguf"):
        logger.info(f"Skip vocab files for model {name}, no GGUF file found")
        continue

    with open(f"models/ggml-vocab-{name}.gguf.inp", "w", encoding="utf-8") as f:
        for text in tests:
            f.write(f"{text}")
            f.write("\n__ggml_vocab_test__\n")

    with open(f"models/ggml-vocab-{name}.gguf.out", "w") as f:
        for text in tests:
            res = tokenizer.encode(text, add_special_tokens=False)
            for r in res:
                f.write(f" {r}")
            f.write("\n")

    logger.info(f"Tests for {name} written in ./models/ggml-vocab-{name}.gguf.*")

# generate commands for creating vocab files

logger.info("\nRun the following commands to generate the vocab files for testing:\n")

for model in models:
    name = model["name"]

    print(f"python3 convert_hf_to_gguf.py models/tokenizers/{name}/ --outfile models/ggml-vocab-{name}.gguf --vocab-only") # noqa: NP100

logger.info("\n")
