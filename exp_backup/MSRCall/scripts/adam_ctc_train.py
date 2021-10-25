# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from ctc.ctc_encoder import Encoder
import generate_dataset.constants as constants
from ctc.opts import add_decoder_args
from ctc.ctc_decoder import BeamCTCDecoder, GreedyDecoder
from ctc.ScheduledOptim import ScheduledOptim
from generate_dataset.train_dataloader import TrainBatchBasecallDataset, TrainBatchProvider
import time
from tqdm import tqdm

from ResNetModule import ResidualBlock, ResidualLSTM
import glob
from utils_PCDARTS import create_exp_dir

# current model: B5a_PyramidFuse012

class simpleRNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(simpleRNN, self).__init__()
        self.conv0 = nn.Conv1d(in_channel, 32, 7, stride=2)
        self.bn0 = nn.BatchNorm1d(32)
        self.tanh0 = nn.Tanh()
        self.relu0 =nn.ReLU()
        self.kernel_size1 = 7
        self.kernel_size2 = 7
        self.padding1 = 3
        self.padding2 = 3
        self.downsample = nn.MaxPool1d(3, stride=2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.convs1 = nn.Conv1d(in_channel, out_channel//4, self.kernel_size1, stride=2, padding=self.padding1)
        self.bns1 = nn.BatchNorm1d(out_channel//4)
        self.convs2 = nn.Conv1d(out_channel//4, out_channel//2, self.kernel_size2, stride=2, padding=self.padding2)
        self.bns2 = nn.BatchNorm1d(out_channel//2)
        self.convs3 = nn.Conv1d(out_channel//2, out_channel//2, self.kernel_size2, stride=2, padding=self.padding2)
        self.bns3 = nn.BatchNorm1d(out_channel//2)
        self.layer1  = ResidualBlock(out_channel//2,out_channel)
        self.layer2  = ResidualBlock(out_channel,out_channel)
        self.layer3  = ResidualBlock(out_channel,out_channel)
        self.layer4  = ResidualBlock(out_channel,out_channel)
        self.layer5  = ResidualBlock(out_channel,out_channel)
        self.lstm1 = nn.LSTM(out_channel//2, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm2 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm3 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm4 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm5 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.reslstm1 = ResidualLSTM(32, out_channel)
        self.reslstm2 = ResidualLSTM(out_channel, out_channel)
        self.reslstm3 = ResidualLSTM(out_channel, out_channel)
        self.reslstm4 = ResidualLSTM(out_channel, out_channel)
        self.reslstm5 = ResidualLSTM(out_channel, out_channel)
        # fwB0 related
        self.lstm01 = nn.LSTM(out_channel//4, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm02 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm03 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm04 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm05 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        # fwB1 related
        self.lstm11 = nn.LSTM(out_channel//2, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm12 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm13 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm14 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm15 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        # fwB2 related
        self.lstm21 = nn.LSTM(out_channel//2, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm22 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm23 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm24 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        self.lstm25 = nn.LSTM(out_channel, out_channel//2, 1, batch_first=False, bidirectional=True)
        ###
        self.fusion   = nn.Conv1d(out_channel*3, out_channel, kernel_size=1, stride=1, bias=False)
        self.globalNorm = nn.Linear(out_channel, 6)# layers.GlobalNormFlipFlop(out_channel, 4, dropRatio=0.5)
        self.layer_norm = nn.LayerNorm(out_channel, eps=1e-6)

    def fwA(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = x.permute(2, 0, 1)
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        # x = self.globalNorm(x)
        return x

    def fwB(self, x):
        x = x.permute(2, 0, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        x = x.permute(1, 2, 0)
        return x

    def fwB0(self, x):
        x = x.permute(2, 0, 1)
        self.lstm01.flatten_parameters()
        x, _ = self.lstm01(x)
        self.lstm02.flatten_parameters()
        x, _ = self.lstm02(x)
        self.lstm03.flatten_parameters()
        x, _ = self.lstm03(x)
        self.lstm04.flatten_parameters()
        x, _ = self.lstm04(x)
        self.lstm05.flatten_parameters()
        x, _ = self.lstm05(x)
        x = x.permute(1, 2, 0)
        return x

    def fwB1(self, x):
        x = x.permute(2, 0, 1)
        self.lstm11.flatten_parameters()
        x, _ = self.lstm11(x)
        self.lstm12.flatten_parameters()
        x, _ = self.lstm12(x)
        self.lstm13.flatten_parameters()
        x, _ = self.lstm13(x)
        self.lstm14.flatten_parameters()
        x, _ = self.lstm14(x)
        self.lstm15.flatten_parameters()
        x, _ = self.lstm15(x)
        x = x.permute(1, 2, 0)
        return x

    def fwB2(self, x):
        x = x.permute(2, 0, 1)
        self.lstm21.flatten_parameters()
        x, _ = self.lstm21(x)
        self.lstm22.flatten_parameters()
        x, _ = self.lstm22(x)
        self.lstm23.flatten_parameters()
        x, _ = self.lstm23(x)
        self.lstm24.flatten_parameters()
        x, _ = self.lstm24(x)
        self.lstm25.flatten_parameters()
        x, _ = self.lstm25(x)
        x = x.permute(1, 2, 0)
        return x

    def fwF(self, x): # still fail
        x = x.permute(2, 0, 1)
        x, _ = self.reslstm1(x)
        x, _ = self.reslstm2(x)
        x, _ = self.reslstm3(x)
        x, _ = self.reslstm4(x)
        x, _ = self.reslstm5(x)
        x = x.permute(1, 2, 0)
        return x

    def fwU(self, x0):
        # x1 = self.U2ConvD1(x)
        # x2 = self.downsample(x1)
        # x2 = self.U2ConvD2(x2)
        # x3 = self.downsample(x2)
        x3 = self.U2ConvD3(x0)
        x4 = self.downsample(x3)
        x4 = self.U2ConvD4(x4)
        x5 = self.downsample(x4)
        x5 = self.U2ConvD5(x5)
        x  = self.UConvT1(x5)
        diff = x4.size()[2] - x.size()[2]
        x  = F.pad(x, [diff//2, diff-diff//2])
        x  = torch.cat([x, x4], dim=1)
        x  = self.U2ConvU1(x)
        x  = self.UConvT2(x)
        diff = x3.size()[2] - x.size()[2]
        x  = F.pad(x, [diff//2, diff-diff//2])
        x  = torch.cat([x, x3], dim=1)
        x  = self.U2ConvU2(x)
        x  = self.UConvT3(x)
        diff = x0.size()[2] - x.size()[2]
        x  = F.pad(x, [diff//2, diff-diff//2])
        x  = torch.cat([x, x0], dim=1)
        x  = self.U2ConvU3(x)
        # x  = self.UConvT4(x)
        # diff = x1.size()[2] - x.size()[2]
        # x  = F.pad(x, [diff//2, diff-diff//2])
        # x  = torch.cat([x, x1], dim=1)
        # x  = self.U2ConvU4(x)
        return x

    def forward(self, x, signal_lengths):
        # x = (bs, 2048, 1)
        x = x.transpose(2, 1)  # (bs, 1, 2048)
        x = self.convs1(x)     # (bs, 64, 1024)
        x = self.bns1(x)
        x0 = self.relu0(x)
        x = self.convs2(x0)     # (bs, 128, 512)
        x = self.bns2(x)
        x1 = self.relu0(x)
        #
        x = self.convs3(x1)     # (bs, 128, 256)
        x = self.bns3(x)
        x2 = self.relu0(x)
        ###############
        x0 = self.fwB0(x0)
        x0 = self.downsample(x0)
        x1 = self.fwB1(x1)
        x2 = self.fwB2(x2)
        x2 = self.upsample(x2)
        x = self.fusion(torch.cat([x0, x1, x2], 1))
        # x should be (bs, 256, 512)
        #####################
        x = x.transpose(2, 1)  # (bs, 512, 256)
        x = self.layer_norm(x)
        x = self.globalNorm(x) # (bs, 512, 6)
        new_signal_lengths = ((signal_lengths + 2 * self.padding1 - self.kernel_size1) / 2 + 1).int()
        new_signal_lengths = ((new_signal_lengths + 2 * self.padding2 - self.kernel_size2) / 2 + 1).int()
        
        return x, new_signal_lengths

def train(model, optimizer, device, opt):
    logfile_name = 'adam_exp_backup/' + opt.save_model + '/train.log'
    
    print(model)
    train_dataset = TrainBatchBasecallDataset(
        signal_dir=opt.train_signal_path, label_dir=opt.train_label_path)
    valid_dataset = TrainBatchBasecallDataset(
        signal_dir=opt.test_signal_path, label_dir=opt.test_label_path)

    list_charcter_error = []
    start = time.time()
    show_shape = True
    for id in range(opt.epoch):
        train_provider = TrainBatchProvider(
            train_dataset, opt.batch_size, shuffle=True)
        valid_provider = TrainBatchProvider(
            valid_dataset, opt.batch_size, shuffle=False)
        # train
        model.train()
        total_loss = []
        batch_step = 0
        target_decoder = GreedyDecoder(
            '-ATCG ', blank_index=0)  # P表示padding，-表示blank
        decoder = BeamCTCDecoder(
            '-ATCG ', cutoff_top_n=6, beam_width=3, blank_index=0)
        while True:
            batch = train_provider.next()
            signal, label = batch
            # added by adam
            # if batch_step > 20:
            #     print('Adam: training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
            #         epoch=id,
            #         step=batch_step,
            #         loss=np.mean(total_loss),
            #         t=(time.time() - start) / 60))
            #     break
            if signal is not None and label is not None:
                batch_step += 1
                if show_shape:
                    with open(logfile_name, 'a') as fptr:
                        fptr.write('signal shape:' + str(signal.size()) + '\n')
                        fptr.write('label shape:' + str(label.size()) + '\n')
                    print('signal shape:', signal.size())
                    print('label shape:', label.size())
                    show_shape = False
                signal = signal.type(torch.FloatTensor).to(
                    device)  # (N,L,C), [32,256,1]
                label = label.type(torch.LongTensor).to(
                    device)  # (N,L), [32,70]

                # forward
                optimizer.zero_grad()
                signal_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                enc_output, enc_output_lengths = model(
                    signal, signal_lengths)  # (N,L,C), [32, 256, 6]
                # print(enc_output.shape) # 32, 512, 6
                # print(enc_output_lengths)

                log_probs = enc_output.transpose(1, 0).log_softmax(
                    dim=-1)  # (L,N,C), [256,32,6]
                assert signal.size(2) == 1
                # input_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                target_lengths = label.ne(constants.PAD).sum(1)

                concat_label = torch.flatten(label)
                concat_label = concat_label[concat_label.lt(constants.PAD)]
                # print('log_probs.shape:    ', log_probs.shape) # 32, 512, 6
                loss = F.ctc_loss(log_probs, concat_label, enc_output_lengths,
                                  target_lengths, blank=0, reduction='sum')
                loss.backward()

                optimizer.step_and_update_lr()
                total_loss.append(loss.item() / signal.size(0))
                if batch_step % opt.show_steps == 0:
                    log_str = 'training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                           epoch=id,
                           step=batch_step,
                           loss=np.mean(total_loss),
                           t=(time.time() - start) / 60)
                    print(log_str)
                    with open(logfile_name, 'a') as fptr:
                        fptr.write(log_str + '\n')
                    start = time.time()
            else:
                log_str = 'training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                       epoch=id,
                       step=batch_step,
                       loss=np.mean(total_loss),
                       t=(time.time() - start) / 60)
                print(log_str)
                with open(logfile_name, 'a') as fptr:
                    fptr.write(log_str + '\n')
                break
        # valid
        start = time.time()
        model.eval()
        total_loss = []
        with torch.no_grad():
            total_wer, total_cer, num_tokens, num_chars = 0, 0, 0, 0
            while True:
                batch = valid_provider.next()
                signal, label = batch
                if signal is not None and label is not None:
                    signal = signal.type(torch.FloatTensor).to(device)
                    label = label.type(torch.LongTensor).to(device)

                    signal_lengths = signal.squeeze(
                        2).ne(constants.SIG_PAD).sum(1)
                    enc_output, enc_output_lengths = model(
                        signal, signal_lengths)

                    log_probs = enc_output.transpose(
                        1, 0).log_softmax(2)  # (L,N,C)

                    assert signal.size(2) == 1
                    # input_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                    target_lengths = label.ne(constants.PAD).sum(1)
                    concat_label = torch.flatten(label)
                    concat_label = concat_label[concat_label.lt(constants.PAD)]

                    loss = F.ctc_loss(log_probs, concat_label, enc_output_lengths, target_lengths, blank=0,
                                      reduction='sum')
                    total_loss.append(loss.item() / signal.size(0))
                    log_probs = log_probs.transpose(1, 0)  # (N,L,C)
                    # print('\nadded by Adam')
                    # print('log_probs: ', log_probs[0, :5, :] )
                    # print('log_probs.shape: ', log_probs.shape )
                    # print('log_probs[0][0].sum(): ', log_probs[0][0].sum() )
                    # print('enc_output_lengths: ', enc_output_lengths[0])
                    # print('enc_output_lengths.len: ', len(enc_output_lengths))
                    target_strings = target_decoder.convert_to_strings(
                        label, target_lengths)
                    decoded_output, _ = decoder.decode(
                        log_probs, enc_output_lengths)
                    # print('target_strings: ', target_strings )
                    # print('decoded_output: ', decoded_output )
                    # assert(False)
                    # assert(True)
                    # decoded_output, _ = target_decoder.decode(
                    #     log_probs, enc_output_lengths)
                    for x in range(len(label)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        cer_inst = decoder.cer(transcript, reference)
                        total_cer += cer_inst
                        num_chars += len(reference)
                else:
                    break
            cer = float(total_cer) / num_chars
            list_charcter_error.append(cer)
            log_str = 'validate: epoch {epoch:d}, loss {loss:.6f}, charcter error {cer:.3f} time: {time:.3f}'.format(
                       epoch=id,
                       loss=np.mean(total_loss),
                       cer=cer * 100,
                       time=(time.time() - start) / 60)
            print(log_str)
            with open(logfile_name, 'a') as fptr:
                fptr.write(log_str + '\n')
            start = time.time()
            if cer <= min(list_charcter_error):
                model_state_dic = model.module.state_dict()
                model_name = 'adam_exp_backup/' + opt.save_model + '/' + opt.save_model + '.chkpt'
                checkpoint = {'model': model_state_dic,
                              'settings': opt,
                              'epoch': id}
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')
                with open(logfile_name, 'a') as fptr:
                    fptr.write('    - [Info] The checkpoint file has been updated.\n')


# python3 ctc_train.py -show_steps 50 -save_model val_val_from_adamModel -from_model adam_model.chkpt -as Holt_val/signals -al Holt_val/labels -es Holt_val/signals -el Holt_val/labels
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_model', help="Save model path", required=True)
    parser.add_argument(
        '-from_model', help="load from exist model", default=None)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-train_signal_path', '-as', required=True)
    parser.add_argument('-train_label_path', '-al', required=True)
    parser.add_argument('-test_signal_path', '-es', required=True)
    parser.add_argument('-test_label_path', '-el', required=True)
    parser.add_argument('-learning_rate', '-lr', default=1e-3, type=float)
    parser.add_argument('-weight_decay', '-wd', default=0.01, type=float)
    parser.add_argument('-warmup_steps', default=10000, type=int)
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_ff', type=int, default=1024)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-label_vocab_size', type=int,
                        default=6)  # {0,1,2,3,4,5}
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-show_steps', type=int, default=200) # Adam: orig: 500
    parser.add_argument('-cuda', default=True)
    opt = parser.parse_args()
    device = torch.device('cuda' if opt.cuda else 'cpu')
    print(device)

    backup_folder = 'adam_exp_backup/' + opt.save_model
    create_exp_dir(backup_folder, scripts_to_save=glob.glob('*.py'))

    if opt.from_model is None:
        # model = Model(d_model=opt.d_model,
        #               d_ff=opt.d_ff,
        #               n_head=opt.n_head,
        #               n_layers=opt.n_layers,
        #               label_vocab_size=opt.label_vocab_size,
        #               dropout=opt.dropout).to(device)
        model = simpleRNN(1, 256).to(device)
        model = nn.DataParallel(model, device_ids=[0,1])

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        optim = torch.optim.Adam(
            model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        optim_schedule = ScheduledOptim(
            optimizer=optim, d_model=opt.d_model, n_warmup_steps=opt.warmup_steps)
    else:
        checkpoint = torch.load(opt.from_model)
        model_opt = checkpoint['settings']
        # use trained model setting cover current setting
        # opt = model_opt
        model = Model(d_model=model_opt.d_model,
                      d_ff=model_opt.d_ff,
                      n_head=model_opt.n_head,
                      n_layers=model_opt.n_layers,
                      label_vocab_size=model_opt.label_vocab_size,
                      dropout=model_opt.dropout).to(device)
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        optim = torch.optim.Adam(
            model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        optim_schedule = ScheduledOptim(
            optimizer=optim, d_model=opt.d_model, n_warmup_steps=opt.warmup_steps)

    train(model=model,
          optimizer=optim_schedule,
          device=device, opt=opt)


if __name__ == "__main__":
    main()

    # class Model(nn.Module):
    #     def __init__(self, d_model, d_ff, n_head, n_layers, label_vocab_size, dropout):
    #         super(Model, self).__init__()
    #         self.encoder = Encoder(d_model=d_model,
    #                                d_ff=d_ff,
    #                                n_head=n_head,
    #                                num_encoder_layers=n_layers,
    #                                dropout=dropout)
    #         self.final_proj = nn.Linear(d_model, label_vocab_size)

    #     def forward(self, signal, signal_lengths):
    #         """
    #         :param signal: a tensor shape of [batch, length, 1]
    #         :param signal_lengths:  a tensor shape of [batch,]
    #         :return:
    #         """
    #         enc_output, enc_output_lengths = self.encoder(
    #             signal, signal_lengths)  # (N,L,C), [32, 256, 256]
    #         out = self.final_proj(enc_output)  # (N,L,C), [32, 256, 6]
    #         return out, enc_output_lengths