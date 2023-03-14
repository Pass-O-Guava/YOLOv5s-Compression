import cv2
import sys
import argparse
import time
import glob

from Processor import Processor
from Visualizer import Visualizer

def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='./weights/yolov5s-simple.trt', help='trt engine file path', required=False)
    parser.add_argument('-i', '--image', default='./data/images/', help='image file path', required=False)
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args.model, letter_box=True)
    visualizer = Visualizer()

    images = glob.glob(f"{args.image}*.jpg")
    print(images)
    for image in images:
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

        output_name = f"./{image.split('/')[-1].split('.')[0]}_{args.model.split('/')[-1].split('.trt')[0]}.jpg"
        visualizer.draw_results(img, pred[:, :4], pred[:, 4], pred[:, 5], output_name)



if __name__ == '__main__':
    main()   
