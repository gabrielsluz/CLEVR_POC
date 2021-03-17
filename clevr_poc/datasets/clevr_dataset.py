import torch
import clevr_poc.utils.logging as logging


logger = logging.get_logger(__name__)


class Clevr(torch.utils.data.Dataset):
    """
    CLEVR dataset class
    __getitem__ returns a tensor (image) and an associated question
    The dataset is indexed by question, not by image.
    The main data structure is a list of dicts. Each dict contains an image path, 
    the question and the answer.
    TODO: How to tokenize the question => use a premade tool
    TODO: Possible answers for CLEVR => get from dataset, and then hardcode
    """

    def __init__(self, mode):
        """
        mode = train, val or test
        """
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Clevrer_des".format(mode)
        self.mode = mode

        logger.info("Constructing Clevrer_des {}...".format(mode))
    
    def __getitem__(self, index):
        return 0


"""
The data is organized into the following directory structure:

images/
  train/
    CLEVR_train_000000.png
    CLEVR_train_000001.png
    [...]
    CLEVR_train_069999.png
  val/
    CLEVR_val_000000.png
    [...]
    CLEVR_val_014999.png
  test/
    CLEVR_test_000000.png
    [...]
    CLEVR_test_014999.png
scenes/
  CLEVR_train_scenes.json
  CLEVR_val_scenes.json
questions/
  CLEVR_train_questions.json
  CLEVR_val_questions.json
  CLEVR_test_questions.json


QUESTION FILE FORMAT
Each of the question files has the following format:

{
  "info": <info>,
  "questions": [<question>]
}

<info> {
  "split": <string: "train", "val", or "test">,
  "version": <string>,
  "date": <string>,
  "license": <string>
}

<question> {
  "split": <string: "train", "val", or "test">,
  "image_index": <integer>,
  "image_filename": <string, e.g. "CLEVR_train_000000.png">,
  "question": <string>,
  "answer": <string>,
  "program": [<function>],
  "question_family_index': <integer>,
}

Answers and programs are omitted from the test data.

<function> {
  "function": <string, e.g. "filter_color">,
  "inputs": [list of integer],
  "value_inputs": [list of strings],
}

Programs are represented as lists of functions. Each function may take as input
both literal values (given in "value_inputs") and output from other functions
(given in "inputs"). Functions are guaranteed to be sorted topologically, so
that j in program[i]['inputs'] if and only if j < i.

As a simple example, consider the question "How many blue cubes are there?"

The program representation for this question would be:

[
  {
    "function": "scene",
    "inputs": [],
    "value_inputs": []
  },
  {
    "function": "filter_color",
    "inputs": [0],
    "value_inputs": ["blue"]
  },
  {
    "function": "filter_shape",
    "inputs": [1],
    "value_inputs": ["cube"]
  },
  {
    "function": "count",
    "inputs": [2],
    "value_inputs": []
  }
]

"""
