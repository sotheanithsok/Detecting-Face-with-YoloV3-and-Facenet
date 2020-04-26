import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
from imageFinder import ImageFinder

def get_args():
    """
    Prase command line input
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=1.0, help='maximum distance for valid images')
    parser.add_argument('--show', default=False, action="store_true", help='show a compilation of detected images')
    parser.add_argument('--full', default=False, action="store_true", help='run algorithm on every group and a range of tau')
    return parser.parse_args()

def normal_run(args):
    """
    Normal run for detect faces belong to a single group
    """
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

def full_run(args):
    """
    Run the face detection algorithm on every group for a given range of tau threshold
    """
    finder = ImageFinder()
    for group in range(1,8,1):
        tau_array = []
        execution_time_array = []
        detected_images_array = []
        recall_array=  []
        precision_array = []
        accuracy_array = []
        metrics_array = []
        for tau in range(0,201,5):
            print("-----------------------------------")
            start = time.time()
            finder.find_images(group, tau/100.0)
            end = time.time()
            print("Group: ", group)
            print("Tau: ", tau/100.0)
            print("Execution time: ", (end - start))
            print("Detected images: ", len(finder.get_detected_images()))
            print("Recall: ", finder.get_recall())
            print("Precision: ", finder.get_precision())
            print("Accuracy:", finder.get_accuracy())
            print("Metrics: (tp: %i,  tn: %i, fp: %i, fn: %i)" %(finder.tp,finder.tn,finder.fp,finder.fn))
            tau_array.append(tau/100.0)
            execution_time_array.append(end-start)
            detected_images_array.append(len(finder.get_detected_images()))
            recall_array.append(finder.get_recall())
            precision_array.append(finder.get_precision())
            accuracy_array.append(finder.get_accuracy())
            metrics_array.append((finder.tp,finder.tn,finder.fp,finder.fn))
        print("tau_array:", tau_array)
        print("execution_time_array:", execution_time_array)
        print("detected_images_array:", detected_images_array)
        print("recall_array:", recall_array)
        print("precision_array:", precision_array)
        print("accuracy_array:", accuracy_array)
        print("metrics_array:", metrics_array)

def _main():
    args = get_args()
    if args.full:
        full_run(args)
    else:
        normal_run(args)
    

if __name__ == "__main__":
    _main()
