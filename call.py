#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename: call.py
# @Date    : 05/07/2019 (base version)
# @Author  : Neng Huang (base version)
# @Email   : csuhuangneng@gmail.com (base version)
# @Date    : 01/01/2022
# @Author  : Yang-Ming Yeh
# @Email   : d05943006@ntu.edu.tw

import argparse
from generate_dataset.trim_raw import trim_and_segment_raw
from statsmodels import robust
import shutil
import os
import copy
import numpy as np
from ctc.ctc_encoder import Encoder
import torch
import torch.nn as nn
import torch.utils.data as Data
import generate_dataset.constants as Constants
from ctc.ctc_decoder import BeamCTCDecoder, GreedyDecoder
from tqdm import tqdm
from multiprocessing import Process, Manager
import time
from ResNetModule import ResidualBlock
from exp_backup.MSRCall.scripts.adam_ctc_train import simpleRNN
# from exp_backup.trial_01.scripts.MSRCall_train import simpleRNN


read_id_list, log_probs_list, output_lengths_list, row_num_list = [], [], [], []
encode_mutex = True
decode_mutex = True


class CallDataset(Data.Dataset):
    def __init__(self, records_dir):
        self.records_dir = records_dir
        self.filenames = os.listdir(records_dir)
        self.count = len(self.filenames)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        signal = np.load(self.records_dir + '/' + fname)
        read_id = os.path.splitext(fname)[0]
        return read_id, signal

#@profile
def encode(model, opt):
    global read_id_list, log_probs_list, output_lengths_list, row_num_list
    manager = Manager()
    encode_mutex = manager.Value('i', 1)
    decode_mutex = manager.Value('i', 1)
    write_mutex = manager.Value('i', 1)

    model.eval()
    call_dataset = CallDataset(opt.records_dir)
    data_iter = Data.DataLoader(
        dataset=call_dataset, batch_size=1, num_workers=0)
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    else:
        shutil.rmtree(opt.output)
        os.makedirs(opt.output)
    outpath = os.path.join(opt.output, 'call.fasta')
    encoded_read_num = 0
    bar_counter = 0
    for batch in tqdm(data_iter):
        bar_counter += 1
        read_id, signal = batch
        read_id = read_id[0]
        signal = signal[0]
        read_id_list.append(read_id)
        signal_segs = signal.shape[0]
        row_num = 0
        encoded_read_num += 1
        batch_size = 256
        while encode_mutex.value != 1:
            time.sleep(0.2)
        for i in range(signal_segs // batch_size + 1):
            if i != signal_segs // batch_size:
                signal_batch = signal[i * batch_size:(i + 1) * batch_size]
            elif signal_segs % batch_size != 0:
                signal_batch = signal[i * batch_size:]
            else:
                continue
            signal_batch = torch.FloatTensor(
                signal_batch).to(opt.device)
            signal_lengths = signal_batch.squeeze(
                2).ne(Constants.SIG_PAD).sum(1)
            output, output_lengths = model(
                signal_batch, signal_lengths)
            
            log_probs = output.log_softmax(2)
            row_num += signal_batch.size(0)
            log_probs_list.append(log_probs.cpu().detach())
            output_lengths_list.append(output_lengths.cpu().detach())
        row_num_list.append(row_num)
        if bar_counter >= 999999:
            print('bar_counter = ', str(bar_counter))
            encode_mutex.value = 0
            decode_process(outpath, encode_mutex, decode_mutex, write_mutex)
            while encode_mutex.value != 1:
                time.sleep(0.2)
            read_id_list[:] = []
            log_probs_list[:] = []
            output_lengths_list[:] = []
            row_num_list[:] = []
            encoded_read_num = 0
        if encoded_read_num == 100:
            encode_mutex.value = 0
            decode_process(outpath, encode_mutex, decode_mutex, write_mutex)
            while encode_mutex.value != 1:
                time.sleep(0.2)
            read_id_list[:] = []
            log_probs_list[:] = []
            output_lengths_list[:] = []
            row_num_list[:] = []
            encoded_read_num = 0
    if encoded_read_num > 0:
        encode_mutex.value = 0
        while decode_mutex.value != 1:
            time.sleep(0.2)
        decode_process(outpath, encode_mutex, decode_mutex, write_mutex)

#@profile
def decode_process(outpath, encode_mutex, decode_mutex, write_mutex):
    global read_id_list, log_probs_list, output_lengths_list, row_num_list
    while decode_mutex.value != 1:
        time.sleep(0.2)
    decode_mutex.value = 0
    probs = torch.cat(log_probs_list)
    lengths = torch.cat(output_lengths_list)
    decode_read_id_list = read_id_list
    decode_row_num_list = row_num_list
    encode_mutex.value = 1
    decoder = BeamCTCDecoder('-ATCG ', blank_index=0, alpha=0.0, lm_path=None, beta=0.0, cutoff_top_n=1, # cutoff_top_n=0
                             cutoff_prob=1.0, beam_width=3, num_processes=8)
    decoded_output, offsets = decoder.decode(probs, lengths)
    idx = 0
    while write_mutex.value != 1:
        time.sleep(0.2)
    fw = open(outpath, 'a')
    write_mutex.value = 0
    for x in range(len(decode_row_num_list)):
        row_num = decode_row_num_list[x]
        read_id = decode_read_id_list[x]
        transcript = [v[0] for v in decoded_output[idx:idx + row_num]]
        idx = idx + row_num
        transcript = ''.join(transcript)
        transcript = transcript.replace(' ', '')
        if len(transcript) > 0:
            fw.write('>' + str(read_id) + '\n')
            fw.write(transcript + '\n')
    fw.close()
    write_mutex.value = 1
    decode_mutex.value = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-records_dir', required=True)
    parser.add_argument('-output', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    argv = parser.parse_args()

    if not os.path.exists(argv.output):
        os.makedirs(argv.output)
    if os.path.exists(os.path.join(argv.output, 'call.fasta')):
        os.remove(os.path.join(argv.output, 'call.fasta'))
    argv.cuda = not argv.no_cuda
    device = torch.device('cuda' if argv.cuda else 'cpu')
    ## if want to use single gpu, set 'argv.device = device' to 'argv.device = 1'
    argv.device = 0 # device

    print('==========================================================')
    print('== Caution: Remember to change the "import" path of model!')
    print('==========================================================')
    specify_single_gpu_id = None
    call_model = simpleRNN(1, 256).to(argv.device)
    # call_model = nn.DataParallel(call_model, device_ids=[0, 1]) # comment this line if load_state_dict key error, and set 'argv.device = device' to 'argv.device = 1'
    checkpoint = torch.load(argv.model)
    call_model.load_state_dict(checkpoint['model'])

    ## if want to transfer to non data parallel model
    ## uncomment next line if you want to test on single GPU
    # specify_single_gpu_id = 1
    if(specify_single_gpu_id is not None):
        argv.device = specify_single_gpu_id
        call_model_2 = simpleRNN(1, 256).to(argv.device)
        call_model_2.load_state_dict(call_model.module.state_dict())
        call_model = call_model_2

    encode(call_model, argv)


if __name__ == "__main__":
    main()

