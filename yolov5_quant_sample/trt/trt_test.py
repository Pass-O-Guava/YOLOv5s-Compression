import cv2
import sys
import argparse
import time

from Processor import Processor
from Visualizer import Visualizer

import os
import glob

def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='./weights/yolov5s-simple.trt', help='trt engine file path', required=False)
    parser.add_argument('-p', '--path', default='./data/images/', help='image file path', required=False)
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args.model, letter_box=True)
    visualizer = Visualizer()

    for image in glob.glob(os.path.join(args.path, '*.jpg')):

        img = cv2.imread(image)

        t1 = time.time()
        
        # inference
        output = processor.detect(img)

        t2 = time.time() - t1
        print(f"==> delay:{t2*1000}ms")
        
        # final results
        pred = processor.post_process(output, img.shape, conf_thres=0.5)

        # print('Detection result: ')
        for item in pred.tolist():
            # print(item)
            pass

        output = os.path.join(args.model.replace(args.model.split('/')[-1], ''), f"result_{image.split('/')[-1]}")
        print(f"==> output: {output}")
        visualizer.draw_results(img, pred[:, :4], pred[:, 4], pred[:, 5], output)



if __name__ == '__main__':
    main()   
