#Takes in command line arguments and runs programs
import sys
from ANN import ANN_model
from CNN import CNN_model
from CNNwithCUDA import CNNwithCUDA_model
def main(model, rebuild_data):
    print("Start!")
    if model == "ANN":
        ANN_model()
    elif model == "CNN":
        CNN_model(rebuild_data)
    elif model == "CNNwC":
        CNNwithCUDA_model()
    else:
        print("Incorrect input")
def arg_handler(arg):
    model = -1
    rebuild_data = 0
    if arg[1] == "0":
        model = "ANN"
    elif arg[1] == "1":
        model = "CNN"
    elif arg[1] == "2":
        model = "CNNwC"
    if arg[2]== "1":
        rebuild_data = 1
    return model, rebuild_data


if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")
    model, rebuild_data = arg_handler(sys.argv)
    main(model, rebuild_data)