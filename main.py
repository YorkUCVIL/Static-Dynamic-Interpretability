import os
import cv2
cv2.setNumThreads(0)
import random
import torch.nn as nn
import config
from utils import *
import csv
from argparse import ArgumentParser
import json
import datetime
import psutil
from tqdm import tqdm

def main():
    print('~~~ Starting dimension estimation! ~~~')
    # load config file
    args = config.load_args()

    # make output folder if it doesn't exist already
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # save args to json file
    with open(args.save_dir + '/' + args.model + '_commandline_args.txt', 'w') as file2:
        json.dump(args.__dict__, file2, indent=2)
    # set device
    device = args.device

    # get model
    print(' > Loading model...')
    print(' > Model: ', args.model)
    args.batch_size = int(args.batch_size)
    if args.stg:
        args.stg = int(args.stg)
        print(' > Stage: ', args.stg)
    print(' > Dataset: ', args.dataset)

    if 'slowfast' in args.model:
        print(' > Using slowfast-{} path'.format(args.path))
        print(' > Using slowfast-fuse: ', args.fuse)


    model = get_model(args)
    model = nn.DataParallel(model)
    model.cuda(device)
    model.eval()

    # get dataset
    print(' > Preparing dataset...')
    dataloader = get_dataloader(args)

    n_iter = math.floor(args.n_examples / args.batch_size)
    print(' > Will process {} examples for {} iterations with a batch size of {}...'.format(args.n_examples, n_iter, args.batch_size))

    # create dict with n_factor lists and factor list
    factor_list = []
    output_dict = {'example1': [],
                   'example2': []}

    print(' > Processing starting...')
    start_time = datetime.datetime.now()
    print(' > Start time: ', start_time)
    # for-loop inference and store values as numpy array
    for i, (factor, example1, example2) in enumerate(tqdm(dataloader)):

        if 'slowfast' in args.model:
            example1, example2 = [example1[0].cuda(device), example1[1].cuda(device)],\
                                 [example2[0].cuda(device), example2[1].cuda(device)]
            output1 = model(x=example1, path=args.path, stage=args.stg, fuse=args.fuse)
            output2 = model(x=example2, path=args.path, stage=args.stg, fuse=args.fuse)
        elif 'timesformer' in args.model:
            example1, example2 = example1.cuda(device), example2.cuda(device)
            output1 = model(x=example1, stage=args.stg)
            output2 = model(x=example2, stage=args.stg)
        else:
            example1, example2 = [example1.cuda(device)], [example2.cuda(device)]
            output1 = model(x=example1, stage=args.stg)
            output2 = model(x=example2, stage=args.stg)

        # pass images through model and get distribution mean
        if len(output1.shape) == 1:
            output1 = output1.unsqueeze(0)
            output2 = output2.unsqueeze(0)

        output1 = Distribution(output1).mode()[0]
        output2 = Distribution(output2).mode()[0]

        # add factor and output to list / array for processing later on
        factor_list.append(factor.detach().cpu().numpy())
        output_dict['example1'].append(output1.detach().cpu().numpy())
        output_dict['example2'].append(output2.detach().cpu().numpy())

        if i == n_iter:
            break


    # dimentionality estimmation
    print(' > Finished processing examples...')
    end_time = datetime.datetime.now()
    print(' > End time: ', end_time)

    print(' > Time taken to process : ', end_time - start_time)

    print(' > Starting Dimentionality Estimation!')
    dims, dims_percent = dim_est(output_dict, factor_list, args)
    print(" >>> Estimated factor dimensionalities: {}".format(dims))
    print(" >>> Ratio to total dimensions: {}".format(dims_percent))

    print('Saving results to {}'.format(args.save_dir + '/' + args.model + '_dim_est.csv'))
    # save to output folder
    with open(args.save_dir + '/' + args.model + '_dim_est.csv', mode='w') as file1:
        writer = csv.writer(file1, delimiter=',')
        writer.writerow(dims)
        writer.writerow(dims_percent)


if __name__ == '__main__':
    main()
