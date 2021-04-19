#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  : 
# @Time    : 2021/4/18 23:26
# @Function: 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



# import


from DataProcessor import *
from Model import *
__version__ = '0.1.0'

# __all__ = ['__version__', 'DataProcessor', 'NerProcessor', 'write_tokens', 'convert_single_example',
           # 'filed_based_convert_examples_to_features', 'file_based_input_fn_builder',
           # 'model_fn_builder', 'train']
__all__ = ['__version__', 'NerProcessor']


logger = set_logger('NER Training')

def train(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

    processors = {
        "ner": NerProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if args.clean and args.do_train:
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

    # check output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    processor = processors[args.ner](args.output_dir, args.input_dir)      # output_dir: BERT vocab的输出路径

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        session_config=session_config
    )

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if args.do_train and args.do_eval:
        # 加载训练数据
        train_examples = processor.get_train_examples()
        num_train_steps = int(
            len(train_examples) *1.0 / args.batch_size * args.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        eval_examples = processor.get_dev_examples()

        # 打印验证集数据信息
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.batch_size)

    label_list = processor.get_labels()
    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
    # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        args=args)

    params = {
        'batch_size': args.batch_size
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    if args.do_train and args.do_eval:
        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(args.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, train_file, args.output_dir)

        # 2.读取record 数据，组成batch
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=True)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        eval_file = os.path.join(args.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            filed_based_convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, eval_file, args.output_dir)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # train and eval togither
        # early stop hook
        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=args.save_checkpoints_steps)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                            hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if args.do_predict:
        token_path = os.path.join(args.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(args.data_dir)
        predict_file = os.path.join(args.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 args.max_seq_length, tokenizer,
                                                 predict_file, args.output_dir, mode="test")

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", args.batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(args.output_dir, "label_test.txt")

        def result_to_pair(writer):
            for predict_line, prediction in zip(predict_examples, result):
                idx = 0
                line = ''
                line_token = str(predict_line.text).split(' ')
                label_token = str(predict_line.label).split(' ')
                len_seq = len(label_token)
                if len(line_token) != len(label_token):
                    logger.info(predict_line.text)
                    logger.info(predict_line.label)
                    break
                for id in prediction:
                    if idx >= len_seq:
                        break
                    if id == 0:
                        continue
                    curr_labels = id2label[id]
                    if curr_labels in ['[CLS]', '[SEP]']:
                        continue
                    try:
                        line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                    except Exception as e:
                        logger.info(e)
                        logger.info(predict_line.text)
                        logger.info(predict_line.label)
                        line = ''
                        break
                    idx += 1
                writer.write(line + '\n')

        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)
        from bert_base.train import conlleval
        eval_result = conlleval.return_report(output_predict_file)
        print(''.join(eval_result))
        # 写结果到文件中
        with codecs.open(os.path.join(args.output_dir, 'predict_score.txt'), 'a', encoding='utf-8') as fd:
            fd.write(''.join(eval_result))
    # filter model
    if args.filter_adam_var:
        if not os.path.exists(os.path.join(args.output_dir, 'checkpoint')):
            logger.info('checkpoint file not exits:'.format(os.path.join(args.output_dir, 'checkpoint')))
            return None
        last_name = None
        with codecs.open(os.path.join(args.output_dir, 'checkpoint'), 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip().split(':')
                if len(line) != 2:
                    continue
                if line[0] == 'model_checkpoint_path':
                    last_name = line[1][2:-1]
                    break
        if last_name is not None:
            sess = tf.Session()
            imported_meta = tf.train.import_meta_graph(os.path.join(args.output_dir, last_name + '.meta'))
            imported_meta.restore(sess, os.path.join(args.output_dir, last_name))
            need_vars = []
            for var in tf.global_variables():
                if 'adam_v' not in var.name and 'adam_m' not in var.name:
                    need_vars.append(var)
            saver = tf.train.Saver(need_vars)
            saver.save(sess, os.path.join(args.output_dir, 'model.ckpt'))


