from copy import deepcopy
import numpy as np
import dynet as dy

def transition_mask(vocab, start, end):
    """Create a mask of illegal moves according to the IOB2 tagging scheme."""
    mask = np.ones((len(vocab), len(vocab)), dtype=np.float32)
    for f in vocab:
        for t in vocab:
            # Don't transition to start
            if t is start:
                mask[vocab[t], vocab[f]] = 0
            # Don't transition from end
            if f is end:
                mask[vocab[t], vocab[f]] = 0
            # Don't start an entity with an I-
            if f is start or f.startswith('O'):
                if t.startswith('I-'):
                    mask[vocab[t], vocab[f]] = 0
            # Don't jump from an entity to an I- of a different type
            if f.startswith('B-') or f.startswith('I-'):
                if t.startswith('I-'):
                    f_type = f.split('-')[1]
                    t_type = t.split('-')[1]
                    if f_type != t_type:
                        mask[vocab[t], vocab[f]] = 0
    return mask


class CRF(object):

    def __init__(self, n_tags, vocab, pc):
        super(CRF, self).__init__()
        pc = pc.add_subcollection(name="crf")
        self.s_idx = n_tags
        self.e_idx = n_tags + 1
        self.n_tags = n_tags + 2
        # Transition x -> y is T[y, x]
        vocab = deepcopy(vocab)
        vocab.w2i['<GO>'] = self.s_idx
        vocab.w2i['<EOS>'] = self.e_idx
        self.mask = transition_mask(vocab, '<GO>', '<EOS>')
        self.inv_mask = (self.mask == 0) * -1e4
        self.transitions = pc.add_parameters((self.n_tags, self.n_tags), name="transition")

    @staticmethod
    def _add_ends(emissions):
        """Add fake emission probs for start and end tags but make them small so they aren't used."""
        return [dy.concatenate([e, dy.inputVector([-1e4, -1e4])], d=0) for e in emissions]

    def _mask(self):
        return dy.cmult(self.transitions, dy.inputTensor(self.mask)) + dy.inputTensor(self.inv_mask)

    def _score_sentence(self, emissions, tags):
        """Calculate the score of a sequence of tags."""
        tags = [self.s_idx] + tags
        score = dy.scalarInput(0)
        masked_t = self._mask()
        for i, e in enumerate(emissions):
            score += dy.pick(dy.pick(masked_t, tags[i + 1]), tags[i]) + dy.pick(e, tags[i + 1])
        score += dy.pick(dy.pick(self.transitions, self.e_idx), tags[-1])
        return score

    def _forward(self, emissions):
        """Calculate the sum of all path scores."""
        alphas = [-1e4] * self.n_tags
        alphas[self.s_idx] = 0
        alphas = dy.inputVector(alphas)
        masked_t = self._mask()
        for emission in emissions:
            add_emission = dy.colwise_add(masked_t, emission)
            scores = dy.colwise_add(dy.transpose(add_emission), alphas)
            scores = dy.cmult(scores, dy.transpose(dy.inputTensor(self.mask)))
            alphas = dy.logsumexp([x for x in scores])
        last_alpha = alphas + dy.pick(self.transitions, self.e_idx)
        alpha = dy.logsumexp([x for x in last_alpha])
        return alpha

    def neg_log_loss(self, emissions, tags):
        """Calculate the CRF loss."""
        emissions = CRF._add_ends(emissions)
        viterbi_score = self._forward(emissions)
        gold_score = self._score_sentence(emissions, tags)
        return viterbi_score - gold_score

    def decode(self, emissions):
        """Generate the best path."""
        emissions = CRF._add_ends(emissions)
        backpointers = []
        masked_t = self._mask()

        alphas = [-1e4] * self.n_tags
        alphas[self.s_idx] = 0
        alphas = dy.inputVector(alphas)

        for emission in emissions:
            next_vars = dy.colwise_add(dy.transpose(masked_t), alphas)
            best_tags = np.argmax(next_vars.npvalue(), 0)
            v_t = dy.max_dim(next_vars, 0)
            alphas = v_t + emission
            backpointers.append(best_tags)

        terminal_expr = alphas + dy.pick(masked_t, self.e_idx)
        best_tag = np.argmax(terminal_expr.npvalue())
        path_score = dy.pick(terminal_expr, best_tag)

        best_path = [best_tag]
        for bp_t in reversed(backpointers):
            best_tag = bp_t[best_tag]
            best_path.append(best_tag)
        _ = best_path.pop()
        return best_path[::-1], path_score
