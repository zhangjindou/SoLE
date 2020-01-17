from collections import defaultdict
import numpy as np
import os


class Result(object):
    def __init__(self, ranks, raw_ranks):
        self.ranks = ranks
        self.raw_ranks = raw_ranks
        self.mrr = np.mean(1.0 / ranks)
        self.raw_mrr = np.mean(1.0 / raw_ranks)

        cnt = float(len(ranks))

        self.hits_at1 = np.sum(ranks <= 1) / cnt
        self.hits_at3 = np.sum(ranks <= 3) / cnt
        self.hits_at10 = np.sum(ranks <= 10) / cnt

        self.raw_hits_at1 = np.sum(raw_ranks <= 1) / cnt
        self.raw_hits_at3 = np.sum(raw_ranks <= 3) / cnt
        self.raw_hits_at10 = np.sum(raw_ranks <= 10) / cnt


class Scorer(object):
    def __init__(self, train, valid, test, n_entities):
        self.known_obj_triples = defaultdict(list)
        self.known_sub_triples = defaultdict(list)
        self.n_entities = n_entities

        self.update_known_triples(train)
        self.update_known_triples(test)
        if valid is not None:
            self.update_known_triples(valid)


    def update_known_triples(self, triples):
        for i, j, k in triples:
            self.known_obj_triples[(i, j)].append(k)
            self.known_sub_triples[(j, k)].append(i)

    def compute_scores(self, predict_func, eval_set, flag=False):
        # preds = predict_func(eval_set)
        fn_pre_result = open("./prediction_results_" + str(os.getpid()), 'w')

        nb_test = len(eval_set)
        ranks = np.empty(2 * nb_test)
        raw_ranks = np.empty(2 * nb_test)

        idx_obj_mat = np.empty((self.n_entities, 3), dtype=np.int64)
        idx_sub_mat = np.empty((self.n_entities, 3), dtype=np.int64)
        idx_obj_mat[:, 2] = np.arange(self.n_entities)
        idx_sub_mat[:, 0] = np.arange(self.n_entities)

        rel_acc = dict()

        def eval_o(i, j):
            idx_obj_mat[:, :2] = np.tile((i, j), (self.n_entities, 1))
            return predict_func(idx_obj_mat)

        def eval_s(j, k):
            idx_sub_mat[:, 1:] = np.tile((j, k), (self.n_entities, 1))
            return predict_func(idx_sub_mat)

        for a, (i, j, k) in enumerate(eval_set):
            res_obj = eval_o(i, j)
            raw_ranks[a] = np.sum(res_obj >= res_obj[k])
            ranks[a] = raw_ranks[a] - np.sum(res_obj[self.known_obj_triples[(i, j)]] >= res_obj[k]) + 1
            ranks[a] = max(1, ranks[a])

            res_sub = eval_s(j, k)
            raw_ranks[nb_test + a] = np.sum(res_sub >= res_sub[i])
            ranks[nb_test + a] = raw_ranks[nb_test + a] - np.sum(
                res_sub[self.known_sub_triples[(j, k)]] >= res_sub[i]) + 1
            ranks[nb_test + a] = max(1, ranks[nb_test + a])

            if flag:
                #print("(%d,%d,%d)\t%d\t%d\t%f" % (i, j, k, ranks[a], ranks[nb_test + a], res_obj[k]))
                fn_pre_result.write("%d\t%d\n" % (a, res_obj[k] >= 0.5))

        rel_acc_sum = 0.0
        for v in rel_acc.values():
            rel_acc_sum += v[0] / v[1]

        fn_pre_result.close()

        return Result(ranks, raw_ranks)

    def compute_ranking(self, predict_func, eval_set):
        nb_test = len(eval_set)

        idx_obj_mat = np.empty((self.n_entities, 3), dtype=np.int64)
        idx_sub_mat = np.empty((self.n_entities, 3), dtype=np.int64)
        idx_obj_mat[:, 2] = np.arange(self.n_entities)
        idx_sub_mat[:, 0] = np.arange(self.n_entities)

        results = list()

        def cmp_second(elem):
            return elem[1]

        def eval_o(i, j):
            idx_obj_mat[:, :2] = np.tile((i, j), (self.n_entities, 1))
            return predict_func(idx_obj_mat)

        def eval_s(j, k):
            idx_sub_mat[:, 1:] = np.tile((j, k), (self.n_entities, 1))
            return predict_func(idx_sub_mat)

        for a, (i, j, k) in enumerate(eval_set):
            obj_set = self.known_obj_triples[(i, j)]
            res_obj = eval_o(i, j)
            obj_score = list()
            for idx, score in enumerate(res_obj):
                #if score >= res_obj[k] and idx not in self.known_obj_triples[(i,j)]:
                if score == res_obj[k] or idx not in obj_set:
                    obj_score.append((idx, score))
            obj_score.sort(key=cmp_second, reverse=True)

            sub_set = self.known_sub_triples[(j, k)]
            res_sub = eval_s(j, k)
            sub_score = list()
            for idx, score in enumerate(res_sub):
                #if score >= res_sub[i] and idx not in self.known_obj_triples[(j, k)]:
                if score == res_sub[i] or idx not in sub_set:
                    sub_score.append((idx, score))
            sub_score.sort(key=cmp_second, reverse=True)

            results.append((a, res_obj[k], obj_score[:5], sub_score[:5]))

        return results

class RelationScorer(object):
    def __init__(self, train, valid, test, n_relations):
        self.known_rel_triples = defaultdict(list)
        self.n_relations = n_relations

        self.update_known_triples(train)
        self.update_known_triples(test)
        if valid is not None:
            self.update_known_triples(valid)

    def update_known_triples(self, triples):
        for i, j, k in triples:
            self.known_rel_triples[(i, k)].append(j)

    def compute_scores(self, predict_func, eval_set):
        # preds = predict_func(eval_set)

        nb_test = len(eval_set)
        ranks = np.empty(nb_test)
        raw_ranks = np.empty(nb_test)

        idx_rel_mat = np.empty((self.n_relations, 3), dtype=np.int64)
        idx_rel_mat[:, 1] = np.arange(self.n_relations)

        def eval_r(i, j):
            idx_rel_mat[:, 0] = i * np.ones(self.n_relations)
            idx_rel_mat[:, 2] = j * np.ones(self.n_relations)
            return predict_func(idx_rel_mat)

        for a, (i, j, k) in enumerate(eval_set):
            res = eval_r(i, k)
            raw_ranks[a] = np.sum(res >= res[j])
            ranks[a] = raw_ranks[a] - np.sum(res[self.known_rel_triples[(i, k)]] >= res[j]) + 1
            ranks[a] = max(1, ranks[a])

        return Result(ranks, raw_ranks)


