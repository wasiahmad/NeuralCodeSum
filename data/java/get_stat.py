import subprocess
from tqdm import tqdm
from prettytable import PrettyTable


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def main():
    records = {'train': 0, 'dev': 0, 'test': 0}
    function_tokens = {'train': 0, 'dev': 0, 'test': 0}
    javadoc_tokens = {'train': 0, 'dev': 0, 'test': 0}
    unique_function_tokens = {'train': set(), 'dev': set(), 'test': set()}
    unique_javadoc_tokens = {'train': set(), 'dev': set(), 'test': set()}

    attribute_list = ["Records", "Function Tokens", "Javadoc Tokens",
                      "Unique Function Tokens", "Unique Javadoc Tokens"]

    def read_data(split):
        source = '%s/code.original_subtoken' % split
        target = '%s/javadoc.original' % split
        with open(source) as f1, open(target) as f2:
            for src, tgt in tqdm(zip(f1, f2),
                                 total=count_file_lines(source)):
                func_tokens = src.strip().split()
                comm_tokens = tgt.strip().split()
                records[split] += 1
                function_tokens[split] += len(func_tokens)
                javadoc_tokens[split] += len(comm_tokens)
                unique_function_tokens[split].update(func_tokens)
                unique_javadoc_tokens[split].update(comm_tokens)

    read_data('train')
    read_data('dev')
    read_data('test')

    table = PrettyTable()
    table.field_names = ["Attribute", "Train", "Valid", "Test", "Fullset"]
    table.align["Attribute"] = "l"
    table.align["Train"] = "r"
    table.align["Valid"] = "r"
    table.align["Test"] = "r"
    table.align["Fullset"] = "r"
    for attr in attribute_list:
        var = eval('_'.join(attr.lower().split()))
        val1 = len(var['train']) if isinstance(var['train'], set) else var['train']
        val2 = len(var['dev']) if isinstance(var['dev'], set) else var['dev']
        val3 = len(var['test']) if isinstance(var['test'], set) else var['test']
        fullset = val1 + val2 + val3
        table.add_row([attr, val1, val2, val3, fullset])

    avg = (function_tokens['train'] + function_tokens['dev'] + function_tokens['test']) / (
            records['train'] + records['dev'] + records['test'])
    table.add_row([
        'Avg. Function Length',
        '%.2f' % (function_tokens['train'] / records['train']),
        '%.2f' % (function_tokens['dev'] / records['dev']),
        '%.2f' % (function_tokens['test'] / records['test']),
        '%.2f' % avg
    ])
    avg = (javadoc_tokens['train'] + javadoc_tokens['dev'] + javadoc_tokens['test']) / (
            records['train'] + records['dev'] + records['test'])
    table.add_row([
        'Avg. Javadoc Length',
        '%.2f' % (javadoc_tokens['train'] / records['train']),
        '%.2f' % (javadoc_tokens['dev'] / records['dev']),
        '%.2f' % (javadoc_tokens['test'] / records['test']),
        '%.2f' % avg
    ])
    print(table)


if __name__ == '__main__':
    main()
