import numpy as np
from utils.data_utils import DataSet
from utils.eval_utils import Scorer, RelationScorer
import os
import config
import tensorflow as tf
from efe import Complex, SoLE_Complex


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Task:
    def __init__(self, model_name, data_name, cv_runs, params_dict, logger, eval_by_rel):
        dataset = DataSet(config.DATASET[data_name])
        self.train_triples, self.valid_triples, self.test_triples = dataset.load_data()
        self.e2id, self.r2id = dataset.load_idx()

        self.groundings = dataset.load_groundings() if "SoLE" in model_name else None
        self.model_name = model_name
        self.data_name = data_name
        self.cv_runs = cv_runs
        self.params_dict = params_dict
        self.hparams = AttrDict(params_dict)
        if "batch_size" not in self.hparams :
            if "batch_num" in self.hparams :
                self.hparams["batch_size"] = int(len(self.train_triples)/self.hparams["batch_num"])
            else:
                raise AttributeError("Need parameter batch_size or batch_num! (Check model_param_space.py)")

        self.logger = logger
        self.n_entities = len(self.e2id)
        self.n_relations = len(self.r2id)
        if eval_by_rel:
            self.scorer = RelationScorer(
                self.train_triples, self.valid_triples, self.test_triples, self.n_relations)
        else:
            self.scorer = Scorer(
                self.train_triples, self.valid_triples, self.test_triples, self.n_entities)

        self.model = self._get_model()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        checkpoint_path = os.path.abspath(config.CHECKPOINT_PATH + "/" + self.__str__() +
                                          ("_NNE_" if "NNE_enable" in self.hparams and self.hparams.NNE_enable == True else "_noNNE_") + data_name)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.checkpoint_prefix = os.path.join(checkpoint_path, self.__str__())

        print(self.hparams)

    def __str__(self):
        return self.model_name

    def _get_model(self):
        args = [self.n_entities, self.n_relations, self.hparams]
        Complex_model_list = ["Complex", "Complex_fb15k", "Complex_db100k", "SoLE_Complex_fb15k", "SoLE_Complex_db100k"]

        if self.model_name in Complex_model_list:
            if "SoLE" in self.model_name:
                return SoLE_Complex(*args)
            else:
                return Complex(*args)
        else:
            raise AttributeError("Invalid model name! (Check model_param_space.py)")

    def _save(self, sess):
        path = self.saver.save(sess, self.checkpoint_prefix)
        print("Saved model to {}".format(path))

    def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info("%s%s:" % (prefix, k))
                self.print_param_dict(v, prefix + incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix, k, v))

    def create_session(self):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=8,
            allow_soft_placement=True,
            log_device_placement=False,
            #gpu_options=gpu_options,
        )
        return tf.Session(config=session_conf)

    def cv(self):
        self.logger.info("=" * 50)
        self.logger.info("Params")
        self._print_param_dict(self.params_dict)
        self.logger.info("Results")
        self.logger.info("\t\tRun\t\tStep\t\tRaw MRR\t\tFiltered MRR")

        cv_res = []
        for i in range(self.cv_runs):
            sess = self.create_session()
            sess.run(tf.global_variables_initializer())
            step, res = self.model.fit(sess, self.train_triples, self.valid_triples, self.scorer, groundings=self.groundings)

            def pred_func(test_triples):
                return self.model.predict(sess, test_triples)

            if res is None:
                step = 0
                res = self.scorer.compute_scores(pred_func, self.valid_triples)
            self.logger.info("\t\t%d\t\t%d\t\t%f\t\t%f" % (i, step, res.raw_mrr, res.mrr))
            cv_res.append(res)
            sess.close()

        self.raw_mrr = np.mean([res.raw_mrr for res in cv_res])
        self.mrr = np.mean([res.mrr for res in cv_res])

        self.raw_hits_at1 = np.mean([res.raw_hits_at1 for res in cv_res])
        self.raw_hits_at3 = np.mean([res.raw_hits_at3 for res in cv_res])
        self.raw_hits_at10 = np.mean([res.raw_hits_at10 for res in cv_res])

        self.hits_at1 = np.mean([res.hits_at1 for res in cv_res])
        self.hits_at3 = np.mean([res.hits_at3 for res in cv_res])
        self.hits_at10 = np.mean([res.hits_at10 for res in cv_res])

        self.logger.info("CV Raw MRR: %.6f" % self.raw_mrr)
        self.logger.info("CV Filtered MRR: %.6f" % self.mrr)
        self.logger.info("Raw: Hits@1 %.3f Hits@3 %.3f Hits@10 %.3f" % (
            self.raw_hits_at1, self.raw_hits_at3, self.raw_hits_at10))
        self.logger.info("Filtered: Hits@1 %.3f Hits@3 %.3f Hits@10 %.3f" % (
            self.hits_at1, self.hits_at3, self.hits_at10))
        self.logger.info("-" * 50)

    def refit(self, if_save=False):
        sess = self.create_session()
        sess.run(tf.global_variables_initializer())
        #self.model.fit(sess, np.concatenate((self.train_triples, self.valid_triples)), groundings=self.groundings)
        self.model.fit(sess, self.train_triples, self.valid_triples, scorer=self.scorer,
                       groundings=self.groundings)
        if if_save:
            self._save(sess)
            #self.model.save_embeddings(sess)

        def pred_func(test_triples):
            return self.model.predict(sess, test_triples)

        #self.scorer.compute_relation_thresholds(pred_func)
        res = self.scorer.compute_scores(pred_func, self.test_triples)
        self.logger.info("Test Results:")
        self.logger.info("Raw MRR: %.6f" % res.raw_mrr)
        self.logger.info("Filtered MRR: %.6f" % res.mrr)
        self.logger.info("Raw: Hits@1 %.3f Hits@3 %.3f Hits@10 %.3f" % (
            res.raw_hits_at1, res.raw_hits_at3, res.raw_hits_at10))
        self.logger.info("Filtered: Hits@1 %.3f Hits@3 %.3f Hits@10 %.3f" % (
            res.hits_at1, res.hits_at3, res.hits_at10))

        sess.close()
        return res

    def restore(self, times=1):
        sess = self.create_session()
        self.saver.restore(sess, self.checkpoint_prefix)

        def pred_func(test_triples):
            return self.model.predict(sess, test_triples)

        #check_triples = [(14574,134,2179),(10110,803,6770),(13133,270,3150),(4676,647,13146)]
        check_triples = list()
        res = self.scorer.compute_scores(pred_func,self.test_triples,True)
        self.logger.info("Selected Test Results:")
        self.logger.info("Raw MRR: %.6f" % res.raw_mrr)
        self.logger.info("Filtered MRR: %.6f" % res.mrr)
        self.logger.info("Raw: Hits@1 %.3f Hits@3 %.3f Hits@10 %.3f" % (
            res.raw_hits_at1, res.raw_hits_at3, res.raw_hits_at10))
        self.logger.info("Filtered: Hits@1 %.3f Hits@3 %.3f Hits@10 %.3f" % (
            res.hits_at1, res.hits_at3, res.hits_at10))

        #rankings = self.scorer.compute_ranking(pred_func,check_triples)
        rankings = list()

        for tri, score, obj_list, sub_list in rankings:
            self.logger.info("Triple (%d,%d,%d), score: %.6f" % (check_triples[tri][0], check_triples[tri][1], check_triples[tri][2], score))
            self.logger.info("\t Top 5 objects:")
            for i, s in obj_list:
                self.logger.info("\t (%d,%.6f)" % (i, s))
            self.logger.info("\t Top 5 subjects:")
            for i, s in sub_list:
                self.logger.info("\t (%d,%.6f)" % (i, s))

        sess.close()
        return
