import fedml
from fedml import FedMLRunner

from model.logisticRegression import LogisticRegression
from model.customModel import CustomModel

import time
from data.data import get_data


if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset= get_data()
    # print("Output Dimensions : " + str(output_dim))
    time.sleep(10)

    # load model
    # model = fedml.model.create(args, output_dim)
    # input_dim=28*28
    # hidden_dim=28*28
    # model=LogisticRegression(input_dim,hidden_dim,output_dim)

    model=CustomModel()

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
