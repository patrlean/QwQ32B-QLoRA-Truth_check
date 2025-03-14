from datasets import load_dataset, Dataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--name", type=str, help="load dataset", default="fava-uw/fava-data", choices=["fava", "tqa"])
parser.add_argument("-p", "--preprocess", type=bool, help="whether to preprocess dataset", default="True")

if __name__ == "__main__":

    args = parser.parse_args()
    if args.name == "fava":
        dataset_name = "fava-uw/fava-data"
        subset = None
    elif args.name == "tqa":
        dataset_name = "truthfulqa/truthful_qa"
        subset = "multiple_choice"


    ds = load_dataset(dataset_name, subset)

    train = ds['validation']

    print(train[0])
