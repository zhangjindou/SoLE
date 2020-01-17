from optparse import OptionParser
from task import Task
import logging
from model_param_space import param_space_dict

def train(model_name, data_name, params_dict, logger, eval_by_rel, if_save, restore_flag):
    task = Task(model_name, data_name, 1, params_dict, logger, eval_by_rel)
    if restore_flag:
        task.restore()
    else:
        task.refit(if_save)

def parse_args(parser):
    parser.add_option("-m", "--model", dest="model_name", type="string", default="best_TransE_L2")
    parser.add_option("-d", "--data", dest="data_name", type="string", default="wn18")
    parser.add_option("-r", "--relation", dest="relation", action="store_true", default=False)
    parser.add_option("-s", "--save", dest="save", action="store_true", default=False)

    parser.add_option("-b", "--restore", dest="restore", action="store_true", default=False)

    options, args = parser.parse_args()
    return options, args

def main(options):
    logger = logging.getLogger()
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
    '''
    train(options.model_name, options.data_name,
        params_dict=param_space_dict[options.model_name],
        logger=logger, eval_by_rel=options.relation, if_save=options.save)
    '''
    train(options.model_name, options.data_name,
          params_dict=param_space_dict[options.model_name],
          logger=logger, eval_by_rel=options.relation, if_save=options.save, restore_flag=options.restore)

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
