from tokenizer.java import java_tokenizer
import argparse
import torch
from c2nl.inputters.constants import DATA_LANG_MAP, LANG_ID_MAP
from c2nl.inputters.utils import process_examples
from main import test
from c2nl.utils.misc import get_project_root
import javalang


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


def set_args(args):
    args.max_src_len = 150
    args.max_tgt_len = 50
    args.code_tag_type = 'original_subtoken'
    args.uncase = True
    args.test_batch_size = 32
    args.data_workers = 5
    args.gamma = 0.0
    args.beta = 0.0
    args.coverage_penalty = 'none'
    args.length_penalty = 'none'
    args.beam_size = 4
    args.n_best = 1
    args.stepwise_penalty = False
    args.block_ngram_repeat = 3
    args.ignore_when_blocking = []
    args.replace_unk = True
    args.only_generate = True


def generate_java_doc_template(file_path, method_hypotheses, method_token_list):
    java_doc = []
    with open(file_path, 'r') as java_file:
        java_file_text = java_file.read()
        tree = javalang.parse.parse(java_file_text)
        method_list = tree.types[0].methods
        for i, method in enumerate(method_list):
            if method.documentation is not None and len(method.documentation) != 0:
                java_doc.append(None)
                continue

            current_method_doc = ['/**\n', ' * ' + method_hypotheses[i][0] + '\n']
            for parameter in method.parameters:
                current_method_doc.append(' * @param ' + parameter.name + '\n')

            current_method_doc.append(' * @return \n')
            current_method_doc.append('*/\n')
            java_doc.append(current_method_doc)

    return java_doc


def get_offset(line):
    first_char_index = len(line) - len(line.lstrip())
    return line[0:first_char_index]


def append_java_doc(file_path, java_doc_temp, method_token_list):
    code_with_doc = []
    with open(file_path, 'r') as java_file:
        current_line_no = 0
        total_method = len(method_token_list)
        current_method_no = 0
        for line in java_file:
            if current_method_no < total_method and current_line_no == method_token_list[current_method_no][0][0] - 1:
                if java_doc_temp[current_method_no] is not None:
                    offset = get_offset(line)
                    for doc_data in java_doc_temp[current_method_no]:
                        code_with_doc.append(offset + doc_data)

                current_method_no = current_method_no + 1

            code_with_doc.append(line)
            current_line_no = current_line_no + 1

    with open(str(get_project_root()) + '/output.java', 'w') as new_doc_file:
        new_doc_file.truncate(0)
        for code in code_with_doc:
            new_doc_file.write(code)


def main(input_args):
    # Set cuda
    input_args.cuda = torch.cuda.is_available()
    input_args.parallel = torch.cuda.device_count() > 1

    set_args(input_args)

    method_token_list = []
    if args.file_type == 'java':
        method_token_list = java_tokenizer.tokenize_java(args.file_path, True)

    if args.file_type == 'method':
        method_token_list.append([[0, 0], java_tokenizer.tokenize_java_method(None, args.file_path)])

    dev_exs = convert_to_token(method_token_list)
    method_hypotheses = test.main(input_args, dev_exs)

    java_doc_temp = generate_java_doc_template(args.file_path, method_hypotheses, method_token_list)

    append_java_doc(args.file_path, java_doc_temp, method_token_list)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Java Code Symmetrization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Adding Java file path argument
    parser.add_argument("-p", "--file_path", help="Input file path", required=True)

    # Adding Java file type argument
    parser.add_argument("-f", "--file_type", help="File type", required=True,
                        choices=['java', 'method'], )

    # Adding Model file path argument
    parser.add_argument("-m", "--model_file", help="Model File path", required=True)

    # Adding Output file path argument
    parser.add_argument("-o", "--pred_file", help="Output File path", required=True)

    # Read arguments from command line
    args = parser.parse_args()

    main(args)
