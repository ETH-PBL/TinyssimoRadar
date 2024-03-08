from pathlib import Path
from argparse import ArgumentParser
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import keras

import dataset
from model import TimeSensitiveCCE, get_tinyssimonn, get_cnn, get_tcn, get_classifier, get_tcn_reduced
from utils import FrameAccuracy, GestureAccuracy, ConfusionMatrix

def get_flops(model:keras.Model) -> int:
    from tensorflow.python.framework.convert_to_constants import (
                convert_variables_to_constants_v2_as_graph,
            )

    # Compute FLOPs for one sample
    inp = model.inputs[0]
    inputs = tf.TensorSpec((1,*inp.shape[1:]), inp.dtype)

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    return flops.total_float_ops

def eval_complexity(model:keras.Model) -> dict:
    return {
        'params': model.count_params(),
        'flops': get_flops(model)
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('mode', choices=('train', 'quantize', 'eval'), nargs='+', help='Mode of operation.')
    parser.add_argument('-d', '--dataset_path', default='dataset', help='Folder containing the dataset.')
    parser.add_argument('-m', '--split_mode', choices=('single-user', 'multi-user', 'leave-one-out'), default='leave-one-out', help='Mode of splitting the dataset.')
    parser.add_argument('-t', '--test_users', nargs='+', action='append', type=str, help='Specify the users to use for testing. Ignored if split_mode is not "leave-one-out".')
    parser.add_argument('-g', '--skip_gesture', nargs='+', action='append', type=str, help="Specify gestures to skip. The name of the gesture must match exactly the name used in the dataset, e.g.: pinch_pinky.")
    parser.add_argument('-b', '--balanced', action='store_true', help='Drops data from person_00 when training with multiple users.')
    parser.add_argument('-a', '--antennas', choices=(1,2,3), default=3, type=int, help='Number of antennas to use.')
    parser.add_argument('-s', '--use_sequence_labels', action='store_true', help='Use sequence labels for training.')

    parser.add_argument('--model_directory', default='models', help='Path where models will be saved. The actual models are saved in a subfolder of this path.')
    parser.add_argument('--model_name', dest='model_path', help='Name of the model subfolder. If not set, models will be saved with currecnt date and time, and loaded from the latest.')

    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging, sweeps and monitoring.')

    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training.')
    parser.add_argument('--cnn_ch', nargs=3, type=int, default=[16,32,16], help='Specify channels for the three layers of the CNN.')
    parser.add_argument('--cnn_groups', type=int, default=1, help='Specify the number of convolution groups in the CNN.')
    parser.add_argument('--tcn_ch', type=int, default=24, help='Specify channels for the TCN.')
    parser.add_argument('--tcn_groups', type=int, default=1, help='Specify the number of convolution groups in the TCN.')
    parser.add_argument('--class_ch', nargs='+', type=int, default=[64,32], help='Specify the units for the 2 hidden layers of the Dense.')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Specify the amount of dropout to apply.')
    parser.add_argument('--ker_reg', type=float, default=0.0, help='Magnitude for L2 weight/kernel regularizer.')
    parser.add_argument('--act_reg', type=float, default=0.0, help='Magnitude for L2 activity regularizer.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and inference.')

    sys.argv = [v.replace("=", " ") for v in sys.argv]
    sys.argv = [v.split() for v in sys.argv]
    sys.argv = sum(sys.argv, [])
    print(sys.argv)
    args = parser.parse_args()

    if args.wandb:
        import wandb
        from wandb.keras import WandbMetricsLogger, WandbEvalCallback, WandbModelCheckpoint
        wandb.init()


    data_dir = Path(args.dataset_path)
    # directory of all models
    model_dir = Path(args.model_directory)
    # directory of the specific model
    if not args.model_path:
        if 'train' in args.mode:
            args.model_path = model_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        else: # eval or quantize
            args.model_path = Path(max(model_dir.iterdir(), key=lambda p: p.stat().st_ctime))
    else:
        args.model_path = Path(model_dir / args.model_path)

    users = dataset.get_people_from_path(data_dir)
    gestures = dataset.get_gestures_from_path(data_dir)
    gestures = gestures if args.skip_gesture is None else [g for g in gestures if g not in [x for y in args.skip_gesture for x in y]]

    if args.split_mode == 'single-user':
        train_users = ['person_00']
        train_data = dataset.load_data(data_dir, train_users, gestures, antennas=args.antennas)
        test_data = train_data.sample(frac=0.15)
        train_data = train_data.drop(test_data.index)

    elif args.split_mode == 'multi-user':
        train_users = users
        train_data = dataset.load_data(data_dir, train_users, gestures, antennas=args.antennas)
        test_data = train_data.sample(frac=0.15)
        train_data = train_data.drop(test_data.index)

    elif args.split_mode == 'leave-one-out':
        args.test_users = ['person_00'] if (args.test_users is None) else [x for y in args.test_users for x in y] # flatten list
        train_users = [u for u in users if u not in args.test_users]
        test_data = dataset.load_data(data_dir, args.test_users, gestures)
        train_data = dataset.load_data(data_dir, train_users, gestures)


    # when balanced is enabled, we drop data from person_00 to keep the dataset balanced (50 samples/gesture/user)
    if(args.balanced and 'person_00' in train_users and len(train_users) > 1):
        print("Droppig 650 samples/gesture from person_00")
        df = train_data[train_data['user'] == 'person_00']
        indexes = [df[df[idx]].head(50).index for idx in df.columns[2:]] # get first 50 samples for each gesture
        indexes = [x for y in indexes for x in y]   # flatten idxes
        train_data = train_data.drop(df.index.difference(indexes)) # drop in the original dataframe

    valid_data = train_data.sample(frac=0.25, random_state=42)
    train_data = train_data.drop(valid_data.index)
    test_gen = dataset.DataGenerator(test_data, args.batch_size, sequence_labels=args.use_sequence_labels)
    train_gen = dataset.DataGenerator(train_data, args.batch_size, sequence_labels=args.use_sequence_labels)
    valid_gen = dataset.DataGenerator(valid_data, args.batch_size, sequence_labels=args.use_sequence_labels)

    print(f"Train data: {len(train_data)} samples")
    print(f"Valid data: {len(valid_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    if 'train' in args.mode:
        print()
        print("################################")
        print("##          TRAINING          ##")
        print("################################")

        sample = train_gen[0]
        x_shape = sample[0].shape[1:]
        y_shape = sample[1].shape[1:]

        print(f"X shape: {x_shape}")
        print(f"Y shape: {y_shape}")

        num_frames = sample[0].shape[1]
        num_labels = sample[1].shape[-1]
        frame_shape = x_shape[1:]

        cnn_cfg = {
            'channels': args.cnn_ch,
            'groups': args.cnn_groups,
            'drop_rate': args.drop_rate,
            'kernel_regularizer': args.ker_reg,
            'activity_regularizer': args.act_reg,
        }
        tcn_cfg = {
            'channels':args.tcn_ch,
            'groups': args.tcn_groups,
            'drop_rate': args.drop_rate,
            'kernel_regularizer': args.ker_reg,
            'activity_regularizer': args.act_reg,
        }
        class_cfg = {
            'units':args.class_ch,
            'drop_rate': args.drop_rate,
            'kernel_regularizer': args.ker_reg,
            'activity_regularizer': args.act_reg,
        }

        model = get_tinyssimonn(sample[0].shape[1:], cnn_cfg=cnn_cfg, tcn_cfg=tcn_cfg, class_cfg=class_cfg, num_classes=num_labels)

        loss = TimeSensitiveCCE() if args.use_sequence_labels else 'categorical_crossentropy'
        metrics = [
            FrameAccuracy(),
            GestureAccuracy(mean=1, name='acc_1f'),
            GestureAccuracy(mean=4, name='acc_4f'),
            ConfusionMatrix(num_labels, 'frames', name='cm_frames'),
            ConfusionMatrix(num_labels, 1, name='cm_1f'),
            ConfusionMatrix(num_labels, 4, name='cm_4f')
            ]
        model.compile(optimizer='adam', loss=TimeSensitiveCCE(), metrics=metrics)

        model.summary(expand_nested=True)

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, min_delta=0.3),]
        if args.wandb:
            callbacks.append(WandbMetricsLogger())
        model.fit(train_gen, batch_size=args.batch_size, epochs=args.epochs, validation_data=valid_gen, callbacks=callbacks)

        print("Evaluating Model")
        model.evaluate(valid_gen)

        cnn = model.get_layer('cnn')
        tcn = model.get_layer('tcn')
        tcn_reduced = get_tcn_reduced(tcn.input.shape[1:], tcn_cfg['channels'], tcn_cfg['groups'], tcn_cfg['drop_rate'], tcn_cfg['kernel_regularizer'], tcn_cfg['activity_regularizer'])
        classifier = model.get_layer('classifier')

        tcn_reduced.set_weights(tcn.get_weights())

        model.save(args.model_path /'full')
        tcn.save(args.model_path / 'tcn')
        tcn_reduced.save(args.model_path / 'tcn_reduced')
        cnn.save(args.model_path / 'cnn')
        classifier.save(args.model_path / 'classifier')

    if 'quantize' in args.mode:
        print()
        print("################################")
        print("##        QUANTIZATION        ##")
        print("################################")
        model = keras.models.load_model(args.model_path / 'full')
        cnn = model.get_layer('cnn')
        tcn = model.get_layer('tcn')
        tcn_reduced = keras.models.load_model(args.model_path / 'tcn_reduced')
        classifier = model.get_layer('classifier')

        # quantize full model
        def representative_dataset_gen():
            n = 100
            for i in range(n if len(train_gen) > n else len(train_gen)):
                batch_x, batch_y = train_gen[i]
                yield [batch_x]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print('Full')
        full_quant_model = converter.convert()

        with open(args.model_path / 'full_quant.tflite', 'wb') as f:
            f.write(full_quant_model)

        quant_debugger = tf.lite.experimental.QuantizationDebugger(
            converter=converter,
            debug_dataset=representative_dataset_gen,
        )
        quant_debugger.run()
        with open(args.model_path / 'quantization_debug.csv', 'w') as f:
            quant_debugger.layer_statistics_dump(f)

        # quantize cnn
        def representative_dataset_gen_cnn():
            n = 100
            for i in range(n if len(train_gen) > n else len(train_gen)):
                batch_x, batch_y = train_gen[i]
                batch_x = batch_x.reshape((-1, *batch_x.shape[2:])) # flatten frames
                yield [batch_x]

        converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen_cnn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print("cnn")
        cnn_quant_model = converter.convert()

        with open(args.model_path / 'cnn_quant.tflite', 'wb') as f:
            f.write(cnn_quant_model)

        # quantize tcn
        def representative_dataset_gen_tcn():
            n = 100
            for i in range(n if len(train_gen) > n else len(train_gen)):
                batch_x, batch_y = train_gen[i]
                old_shape = batch_x.shape
                batch_x = batch_x.reshape((-1, *batch_x.shape[2:])) # flatten frames
                batch_x = cnn.predict(batch_x, verbose=0)
                batch_x = batch_x.reshape((-1, old_shape[1], *batch_x.shape[1:]))
                yield [batch_x]

        converter = tf.lite.TFLiteConverter.from_keras_model(tcn)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen_tcn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print("tcn")
        tcn_quant_model = converter.convert()

        with open(args.model_path / 'tcn_quant.tflite', 'wb') as f:
            f.write(tcn_quant_model)

        converter = tf.lite.TFLiteConverter.from_keras_model(tcn_reduced)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen_tcn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print("tcn_reduced")
        tcn_quant_model = converter.convert()

        with open(args.model_path / 'tcn_reduced.tflite', 'wb') as f:
            f.write(tcn_quant_model)

        # quantize classifier
        def representative_dataset_gen_classifier():
            n = 100
            for i in range(n if len(train_gen) > n else len(train_gen)):
                batch_x, batch_y = train_gen[i]
                old_shape = batch_x.shape
                batch_x = batch_x.reshape((-1, *batch_x.shape[2:])) # flatten frames
                batch_x = cnn.predict(batch_x, verbose=0)
                batch_x = batch_x.reshape((-1, old_shape[1], *batch_x.shape[1:]))
                batch_x = tcn.predict(batch_x, verbose=0)
                batch_x = batch_x.reshape((-1, *batch_x.shape[2:])) # flatten frames
                yield [batch_x]

        converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen_classifier
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print("classifier")
        classifier_quant_model = converter.convert()

        with open(args.model_path / 'classifier_quant.tflite', 'wb') as f:
            f.write(classifier_quant_model)

    if 'eval' in args.mode:
        print()
        print("################################")
        print("##         EVALUATION         ##")
        print("################################")
        model = keras.models.load_model(args.model_path / 'full')
        cnn = keras.models.load_model(args.model_path / 'cnn')
        tcn = keras.models.load_model(args.model_path / 'tcn')
        tcn_reduced = keras.models.load_model(args.model_path / 'tcn_reduced')
        classifier = keras.models.load_model(args.model_path / 'classifier')

        model.evaluate(test_gen, verbose=0) # needed to initialize the metrics

        x_test, y_true = test_gen.get_data()
        y_pred = model.predict(x_test, verbose=0)

        complexity = {
            'full': eval_complexity(model),
            'cnn': eval_complexity(cnn),
            'tcn': eval_complexity(tcn),
            'tcn_reduced': eval_complexity(tcn_reduced),
            'classifier': eval_complexity(classifier),
        }
        complexity['runtime'] = {
            'params': sum((complexity['cnn']['params'], complexity['tcn_reduced']['params'], complexity['classifier']['params'])),
            'flops': sum((complexity['cnn']['flops'], complexity['tcn_reduced']['flops'], complexity['classifier']['flops'])),
        }
        if args.wandb:
            wandb.run.summary['complexity'] = complexity
        print("Network complexity:", complexity)

        for m in model.metrics:
            m.reset_state()
            m.update_state(y_true, y_pred)
            if args.wandb:
                if 'cm' in m.name:
                    wandb.run.summary[f"eval/float32_{m.name}"] = m.result().numpy().tolist()
                else:
                    wandb.run.summary[f"eval/float32_{m.name}"] = m.result()
                # np.savetxt(args.model_path / f"float32_{m.name}.csv", m.result(), fmt="%d",
                #             delimiter=',', comments='', header=','.join(gestures))
            print(f"eval/float32_{m.name}:\n{m.result()}")

        if Path(args.model_path / 'full_quant.tflite').exists():
            with open(args.model_path / 'full_quant.tflite', 'rb') as f:
                full_quant_model = f.read()

            interpreter = tf.lite.Interpreter(model_content=full_quant_model, num_threads=8)
            inp = interpreter.get_input_details()[0]['index']
            out = interpreter.get_output_details()[0]['index']
            interpreter.resize_tensor_input(inp, x_test.shape, strict=True)
            interpreter.allocate_tensors()

            quant_scale, quant_zero = interpreter.get_input_details()[0]['quantization']

            x_test = (x_test/quant_scale + quant_zero).astype(np.int8)

            interpreter.set_tensor(inp, x_test)
            interpreter.invoke()
            y_pred = interpreter.get_tensor(out)

            for m in model.metrics:
                m.reset_state()
                m.update_state(y_true, y_pred)
                print(f"Monolitic {m.name}:\n{m.result()}")

            # separate quantization
            with open(args.model_path / 'cnn_quant.tflite', 'rb') as f:
                cnn_quant_model = f.read()
            with open(args.model_path / 'tcn_quant.tflite', 'rb') as f:
                tcn_quant_model = f.read()
            with open(args.model_path / 'classifier_quant.tflite', 'rb') as f:
                classifier_quant_model = f.read()

            cnn_interpreter = tf.lite.Interpreter(model_content=cnn_quant_model, num_threads=8)
            tcn_interpreter = tf.lite.Interpreter(model_content=tcn_quant_model, num_threads=8)
            classifier_interpreter = tf.lite.Interpreter(model_content=classifier_quant_model, num_threads=8)

            cnn_inp = cnn_interpreter.get_input_details()[0]['index']
            cnn_out = cnn_interpreter.get_output_details()[0]['index']
            tcn_inp = tcn_interpreter.get_input_details()[0]['index']
            tcn_out = tcn_interpreter.get_output_details()[0]['index']
            classifier_inp = classifier_interpreter.get_input_details()[0]['index']
            classifier_out = classifier_interpreter.get_output_details()[0]['index']

            cnn_input_shape = cnn_interpreter.get_input_details()[0]['shape']
            cnn_interpreter.resize_tensor_input(cnn_inp, (x_test.shape[0], *cnn_input_shape[1:]), strict=True)
            tcn_input_shape = tcn_interpreter.get_input_details()[0]['shape']
            tcn_interpreter.resize_tensor_input(tcn_inp, (x_test.shape[0], *tcn_input_shape[1:]), strict=True)
            classifier_input_shape = classifier_interpreter.get_input_details()[0]['shape']
            classifier_interpreter.resize_tensor_input(classifier_inp, (x_test.shape[0], *classifier_input_shape[1:]), strict=True)

            cnn_interpreter.allocate_tensors()
            tcn_interpreter.allocate_tensors()
            classifier_interpreter.allocate_tensors()

            frames = []
            for i in range(x_test.shape[1]):
                frame = x_test[:,i]
                cnn_interpreter.set_tensor(cnn_inp, frame)
                cnn_interpreter.invoke()
                y = cnn_interpreter.get_tensor(cnn_out)
                frames.append(y)
            frames = np.stack(frames, axis=1)

            tcn_interpreter.set_tensor(tcn_inp, frames)
            tcn_interpreter.invoke()
            x = tcn_interpreter.get_tensor(tcn_out)

            frames = []
            for i in range(x.shape[1]):
                frame = x[:,i]
                classifier_interpreter.set_tensor(classifier_inp, frame)
                classifier_interpreter.invoke()
                y = classifier_interpreter.get_tensor(classifier_out)
                frames.append(y)
            y_pred = np.stack(frames, axis=1)

            for m in model.metrics:
                m.reset_state()
                m.update_state(y_true, y_pred)

                if args.wandb:
                    if 'cm' in m.name:
                        wandb.run.summary[f"eval/int8_{m.name}"] = m.result().numpy().tolist()
                    else:
                        wandb.run.summary[f"eval/int8_{m.name}"] = m.result()
                    # np.savetxt(args.model_path / f"int8_{m.name}.csv", m.result(), fmt="%d",
                    #             delimiter=',', comments='', header=','.join(gestures))
                print(f"eval/int8_{m.name}:\n{m.result()}")

    if args.wandb:
        wandb.finish()
