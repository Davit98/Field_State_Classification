import os

import numpy as np

import torch
import torch.nn.functional as F

from torch import max, flatten

from .models import MyResNet

PROCESSED_DATA_PATH = 'processed_data/'

label_encoding_reverse = {
	0 : 'Canopy Closed',
	1 : 'Not Planted',
	2 : 'Emerged',
	3 : 'Planted',
	4 : 'Harvested',
	5 : 'Drydown'
}

def test(TRAINED_MODEL,SAMPLE_IMG_NAME):
	'''
	Use already trained model to make prediction for a single sample.

	Parameters
	----------
	TRAINED_MODEL : str
		Path of .pt file for ResNet-18 model.

	SAMPLE_IMG_NAME : str
		Name of a test image (e.g. 3KR1212BR_48_test.npy).
	'''

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = MyResNet()
	net.to(device)

	if torch.cuda.is_available():
		net.load_state_dict(torch.load(TRAINED_MODEL))
	else:
		net.load_state_dict(torch.load(TRAINED_MODEL,map_location=torch.device('cpu')))
	net.eval()

	sample = np.load(os.path.join(PROCESSED_DATA_PATH, SAMPLE_IMG_NAME))
	sample = torch.from_numpy(sample).float().permute(2,0,1).unsqueeze(0)
	sample = sample.to(device)

	_, predicted = max(F.softmax(net(sample), dim=1).data, 1)

	print('Predicted label = {}'.format(label_encoding_reverse[predicted.item()]))


