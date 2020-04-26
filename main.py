import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
from imageFinder import ImageFinder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=0.75, help='maximum distance for valid images')
    parser.add_argument('--show', default=False, action="store_true", help='show a compilation of detected images')
    return parser.parse_args()

def _main():
    args = get_args()
    done = False
    finder = ImageFinder()

    while not done:
        try:
            print("-----------------------------------")
            group = input("(Type \"done\" to exit program) Enter group: ")

            if str(group) == "done":
                done=True
                break

            group = int(group)
            start = time.time()
            finder.find_images(group, args.tau)
            end = time.time()
            print("Group: ", group)
            print("Tau: ", args.tau)
            print("Execution time: ", (end - start))
            print("Detected images: ", len(finder.get_detected_images()))
            print("Recall: ", finder.get_recall())
            print("Precision: ", finder.get_precision())
            print("Accuracy:", finder.get_accuracy())
            print("Metrics: (tp: %i,  tn: %i, fp: %i, fn: %i)" %(finder.tp,finder.tn,finder.fp,finder.fn))
            if(args.show):
                finder.get_detected_images_as_one().show()
        except:
            print("Invalid input")

if __name__ == "__main__":
    _main()
