from c2nl.tokenizers.code_tokenizer import CodeTokenizer, Tokens, Tokenizer
import argparse
import re
from os import path
import javalang
from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.parent)


def get_java_method_map(tree):
    """High level model that handles initializing the underlying network
        architecture, saving, updating examples, and predicting examples.
        """
    method_map = []
    for method in tree.types[0].methods:
        if len(method.annotations) > 0:
            method_map.append([method.annotations[0].position.line, method.position.line])
        else:
            method_map.append([method.position.line, method.position.line])

    return method_map


def get_method_location_map(java_file_path):
    method_map = []
    with open(java_file_path, 'r') as java_file:
        java_file_text = java_file.read()
        tree = javalang.parse.parse(java_file_text)
        method_map = get_java_method_map(tree)

    return method_map


def process_java_file(java_file_path):
    method_map = get_method_location_map(java_file_path)
    total_methods = len(method_map)

    method_text = []
    tokenizer = CodeTokenizer(True, True)
    with open(java_file_path, 'r') as process_sample_file:
        current_line_no = 1
        method_no = 0
        current_method = []
        count_open_bracket = 0
        count_close_bracket = 0
        verify = False
        for x in process_sample_file:
            if current_line_no >= method_map[method_no][0]:
                current_method.append(x)

            if current_line_no >= method_map[method_no][1]:
                count_open_bracket = count_open_bracket + x.count('{')
                count_close_bracket = count_close_bracket + x.count('}')
                if count_open_bracket > 0:
                    verify = True

            if count_open_bracket == count_close_bracket and verify:
                temp_method_text = ' '.join([line.strip() for line in current_method])
                temp_method_text = tokenize_java_method(tokenizer, temp_method_text)
                method_text.append([method_map[method_no], temp_method_text])
                current_method = []
                method_no = method_no + 1
                count_open_bracket = 0
                count_close_bracket = 0
                verify = False
                if method_no == total_methods:
                    break

            current_line_no = current_line_no + 1
        method_text.pop(0)

    return method_text


def tokenize_java_method(tokenizer, inline_method_text):
    if tokenizer is None:
        tokenizer = CodeTokenizer(True, True)

    text = ''
    for i in tokenizer.tokenize(inline_method_text).data:
        s = '(@|\+|\-|,|\]|\[|{|}|=|!|\(|\)|>|<|;|"|/|\.)'
        res = list(filter(None, re.split(s, str(i[0]))))
        res = ' '.join(res)
        text = text + ' ' + res
    return text[1:]


def tokenize_java(java_file_path, save_data):
    # check if the file exist
    if path.exists(java_file_path):
        print("Processing the java file : % s" % java_file_path)
    else:
        raise Exception('No such java file at location: %s' % java_file_path)

    method_text = process_java_file(java_file_path)

    if save_data:
        with open(get_project_root() + '/output.code', 'w+') as sample_file:
            for line, method in method_text:
                sample_file.write(method + '\n')
        print('Saving tokenize fine into : %s' % get_project_root() + '/output.code')
    return method_text


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Java Code Tokenizer Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Adding Java file path argument
    parser.add_argument("-p", "--file_path", help="Input file path", required=True)

    # Adding Java file path argument
    parser.add_argument("-f", "--file_type", help="File type", required=True,
                        choices=['java', 'method'], )

    # Read arguments from command line
    args = parser.parse_args()

    if args.file_type == 'java':
        print("Tokenized : % s" % tokenize_java(args.file_path, True))

    if args.file_type == 'method':
        if path.exists(args.file_path):
            print("Processing the file : % s" % args.file_path)
            with open(args.file_path, 'r') as sample_file:
                java_file_content = sample_file.read()
                tokenize_method_text = tokenize_java_method(None, java_file_content)
                with open('../../output.code', 'w+') as output_file:
                    output_file.write(tokenize_method_text)
                print("Tokenized : % s" % tokenize_method_text)
        else:
            raise Exception('No such file at location: %s' % args.file_path)
