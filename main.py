import json

import test
import train
from dataproc import Dataflow
from models import neuralNets

json_path = 'dataproc/dataset/'  # TODO use argparse
json_name = 'dataset.json'

f = open(json_path + json_name, )  # Load Json dataset
data_dict = json.load(f)
f.close()

dataset = Dataflow.DataFlow(data_dict=data_dict)
train_data, test_data = dataset.train, dataset.test  # dataset conversion for training, testing

net = neuralNets.neuralNets().model  # call deepXRD with default parameters

train.Trainer(net=net,
              data=train_data)  # call trainer and fit data to model

predictions = test.Tester(net=net,
                          data=test_data).preds_cnn  # predict for test dataset, and then do further analysis as needed
