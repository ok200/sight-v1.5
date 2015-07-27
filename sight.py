#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.

 python sight.py --raw_scale 256 ./me.jpg  ./result.npy --gpu

*requirement caffe ios


"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import caffe

import cv2
import liblo

def main(argv):
    cap = cv2.VideoCapture(0)

    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "caffe/models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    mean = np.load('caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    # args.input_file = os.path.expanduser(args.input_file)
    # if args.input_file.endswith('npy'):
    #     print("Loading file: %s" % args.input_file)
    #     inputs = np.load(args.input_file)

    # elif os.path.isdir(args.input_file):
    #     print("Loading folder: %s" % args.input_file)
    #     inputs =[caffe.io.load_image(im_f)
    #              for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    # else:
    #     print("Loading file: %s" % args.input_file)
    #     inputs = [caffe.io.load_image(args.input_file)]


    categories = np.loadtxt('caffe/data/ilsvrc12/synset_words.txt', str, delimiter="\t")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # cv2.imshow('frame', frame)
        inputs = [frame]
        # inputs = [caffe.io.load_image('bearskin.jpg')]

        # Classify.
        # start = time.time()
        predictions = classifier.predict(inputs, True)
        # print("Done in %.2f s." % (time.time() - start))
        prediction = zip(predictions[0].tolist(), range(len(predictions[0])), categories)
        prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
        top_k = 3

        param_table = [[0.2, 0.5, 0.8]] * 10
        msg_args = [top_k]
        target = liblo.Address("127.0.0.1", 12000)

        for rank, (score, cat_no, cat) in enumerate(prediction[:top_k], start=1):
            params = [param_table[i][x] for i, x in enumerate([(cat_no / pow(3, i)) % 3 for i in range(10)])]
            print(score, cat, params)
            msg_args.append(20 * score)
            msg_args.append(320.0)
            msg_args.append(320.0)
            msg_args += params
        liblo.send(target, "/frame", *msg_args)

    # Save
    print("Saving results into %s" % args.output_file)
    np.save(args.output_file, predictions)


if __name__ == '__main__':
    main(sys.argv)
