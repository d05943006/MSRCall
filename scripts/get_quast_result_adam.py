import os
import pandas as pd
import sys

def get_report(root_path):
    # root_path = "adam_results/assemblies/flye_assemblies/SACall_AP_quast_report.tsv"
    tmp_dic = {}
    with open(root_path) as fpath:
        for line in fpath:
            if any([v in line for v in ['N50', '# misassemblies', 'NA50', '# mismatches per 100 kbp', '# indels per 100 kbp', 'Genome fraction (%)']]):
                [key, value] = line.strip().split('\t')
                assert key in ['N50', '# misassemblies', 'NA50',
                                '# mismatches per 100 kbp', '# indels per 100 kbp', 'Genome fraction (%)'], print(key)
                tmp_dic[key] = value

    tmp_str = ''
    index=['# misassemblies', '# mismatches per 100 kbp', '# indels per 100 kbp', 'N50',  'NA50', 'Genome fraction (%)']
    index_str='# misassemblies,    # mismatches per 100 kbp,  # indels per 100 kbp,     N50,                    NA50,         Genome fraction (%)'
    for idx in index:
        if(tmp_str==''):
            tmp_str = '       ' + str(tmp_dic[idx])
        else:
            tmp_str = tmp_str + ',           \t' + str(tmp_dic[idx])

    modelName = root_path.split('/')[-1]
    print(modelName[:-17])
    # print(index_str)
    print(tmp_str)


def main():
    get_report(sys.argv[1])


if __name__ == "__main__":
    main()

