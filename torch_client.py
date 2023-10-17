import fedml
from fedml import FedMLRunner

from model.logisticRegression import LogisticRegression


if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)
    print("Output Dimensions : " + str(output_dim))

    # load model
    # model = fedml.model.create(args, output_dim)
    input_dim=28*28
    hidden_dim=128
    model=LogisticRegression(input_dim,hidden_dim,output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
