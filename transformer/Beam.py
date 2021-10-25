import torch
import numpy as np
import transformer.constants as Constants


class Beam(object):
    ''' Store the neccesary info for beam search. '''

    def __init__(self, size, device):

        self.size = size
        self.done = False

        self.device = device

        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).zero_().to(device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.LongTensor(size).fill_(Constants.PAD).to(device)]
        for i in range(size):
            self.next_ys[0][i] = Constants.BOS
        self.bestpath = [[Constants.BOS]]*size

    def get_current_state(self):
        "Get the outputs for the current timestep."
        # print('####'*20)
        # print('best path:\n',self.bestpath)
        # print('get_current_state:\n',self.get_tentative_hypothesis())
        # assert torch.LongTensor(self.bestpath).cuda().equal(self.get_tentative_hypothesis().cuda())
        # print('****' * 20)
        return torch.LongTensor(self.bestpath)

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_lk):
        # word_lk is the prediction of the current instance at current time step
        "Update the status and check for finished or not."
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            # print('score:\n',self.scores)
            # print('word_lk:\n',word_lk)
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        # print('beam_lk:\n',beam_lk)
        flat_beam_lk = beam_lk.view(-1)
        # print('flat_beam_lk:\n',flat_beam_lk)

        # best_scores, best_scores_id = flat_beam_lk.topk(
        #     self.size, 0, True, True)  # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(
            self.size, 0, True, True)  # 2nd sort
        # print('best_scores:\n',best_scores)
        # print('best_scores_id:\n',best_scores_id)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        # print('prev_k:\n', prev_k)
        # if prev_k.sum().item()!=3:
        #     print('!\n'*100)
        self.prev_ks.append(prev_k)
        ys = best_scores_id - prev_k * num_words
        # print('next_ys:\n',ys)
        # (best_scores_id - prev_k * num_words) is the index of the best vocab in each Beam
        self.next_ys.append(ys)
        bstpath = []
        for bst_i in range(len(prev_k)):
            bstpath.append(self.bestpath[prev_k[bst_i].item()]+[ys[bst_i].item()])
        self.bestpath = bstpath
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == Constants.EOS:
            self.done = True
            self.all_scores.append(self.scores)

        return self.done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            # print('hyps:\n',hyps)
            hyps = [[Constants.BOS] + h for h in hyps]
            dec_seq = torch.from_numpy(np.array(hyps))

        return dec_seq

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.
        Parameters.
             * `k` - the position in the beam to construct.
         Returns.
            1. The hypothesis
            2. The attention at each time step.
        """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return hyp[::-1]