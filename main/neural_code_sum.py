from tokenizer.java import java_tokenizer
import argparse
import torch
from c2nl.inputters.constants import DATA_LANG_MAP, LANG_ID_MAP
from c2nl.inputters.utils import process_examples
from main import test
from c2nl.utils.misc import get_project_root
import json


def convert_to_token(sources):
    examples = []
    lang = "java"
    for index, src in sources:
        _ex = process_examples(LANG_ID_MAP[DATA_LANG_MAP[lang]],
                               src,
                               None,
                               None,
                               args.max_src_len,
                               args.max_tgt_len,
                               args.code_tag_type,
                               uncase=args.uncase)
        if _ex is not None:
            examples.append(_ex)

    return examples


def add_project_path(input_args):
    project_root = str(get_project_root()) + '/'
    input_args.model_file = project_root + input_args.model_file
    input_args.pred_file = project_root + input_args.pred_file


def main(input_args, file_path, file_type):
    # Set cuda
    input_args.cuda = torch.cuda.is_available()
    input_args.parallel = torch.cuda.device_count() > 1

    add_project_path(input_args)

    method_token_list = []
    if file_type == 'java':
        method_token_list = java_tokenizer.tokenize_java(str(get_project_root()) + '/' + file_path, True)

    if file_type == 'method':
        method_token_list.append([[0, 0], java_tokenizer.tokenize_java_method(None, file_path)])

    dev_exs = convert_to_token(method_token_list)
    print(test.main(input_args, dev_exs))


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Java Code Symmetrization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Adding Java file path argument
    parser.add_argument("-p", "--file_path", help="Input file path", required=True)

    # Adding Java file path argument
    parser.add_argument("-f", "--file_type", help="File type", required=True,
                        choices=['java', 'method'], )
    # Read arguments from command line
    args = parser.parse_args()
    file_path_arg = args.file_path
    file_type_arg = args.file_type
    with open('commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    main(args, file_path_arg,file_type_arg)
