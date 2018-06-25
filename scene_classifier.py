import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from skimage import io, transform
import pdb
import csv

class SceneClassificationDataset:
    
    def __init__(self, root_dir):
	self.root_dir = os.path.join(root_dir, 'frame')
	self.label_dir = os.path.join(root_dir, 'pose')
#	self.__getitem__()

    def __len__(self):
	return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
	data = os.listdir(self.root_dir)
	label_data = os.listdir(self.label_dir)

	label_name = os.path.join(self.label_dir, label_data[idx])
	position=np.loadtxt(label_name, usecols=[0])
	if position[1] <= -30:
	    label = np.array( [0] )
	elif position[1] > -30 and position[1] <= -15: #-20:
	    label = np.array( [1] )
	elif position[1] > -15 and position[1] <= 0:
#	elif position[1] > -20 and position[1] <= -10:
	    label = np.array( [2] )
#	elif position[1] > -10 and position[1] <= 0:
#	    label = np.array( [3] )
	else:
	    label = np.array( [3] )

	img_name = os.path.join(self.root_dir, data[idx])
#	print img_name
	image = io.imread(img_name)
	image = image.transpose((2, 0, 1))
	sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}
	return sample
	
#	print label_name
#	print img_name

class Net(nn.Module):
    def __init__(self):
	super(Net, self).__init__()
	self.conv1 = nn.Conv2d(3,6,5)
	self.pool = nn.MaxPool2d(2,2)
	self.conv2 = nn.Conv2d(6,16,5)
	self.fc1 = nn.Linear(16 * 72 * 72, 120)
	self.fc2 = nn.Linear(120, 84)
	self.fc3 = nn.Linear(84, 4)
	self.sm = nn.Softmax(1)

    def forward(self, x):
	x = self.pool(F.relu(self.conv1(x)))
	x = self.pool(F.relu(self.conv2(x)))
#	print x.size()
	x = x.view(-1, 16*72*72)
	x = F.relu(self.fc1(x))
	x = F.relu(self.fc2(x))
	x = self.fc3(x)
	return x
#	return self.sm(x)

torch.set_num_threads(4)
scene_dataset = SceneClassificationDataset(root_dir='training_dataset')
#sample = scene_dataset[1]
#print (sample['image'].size(), sample['label'].size())

dataloader = DataLoader(scene_dataset, batch_size=20, shuffle=True, num_workers=1)
#print dataloader
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3) #, momemtum=0.9)
print "Finished setup"

writer = csv.writer(open("loss.csv", 'w'))
for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
#	inputs, labels = data
	optimizer.zero_grad()
	outputs = net(data['image'].type(torch.FloatTensor))
#	print len(outputs)
#	pdb.set_trace()
	loss = criterion(outputs, data['label'][:,0].type(torch.LongTensor))
	loss.backward()
	optimizer.step()

	running_loss += loss.item()
	if i%10 == 9:
	    print('[%d, %5d] loss: %.3f' %(epoch + 1, i+1, running_loss / 10))
	    writer.writerow([running_loss/10])
	    running_loss = 0.0

print "Finished training"
	
predictions = csv.writer(open("predictions.csv", 'w'))
testlabels  = csv.writer(open("labels.csv", 'w'))
test_dataset = SceneClassificationDataset(root_dir='test_dataset')
testdataloader = DataLoader(test_dataset, batch_size=20, shuffle=True, num_workers=1)
correct = 0
total = 0
with torch.no_grad():
    for testdata in testdataloader:
	output = net(testdata['image'].type(torch.FloatTensor))
	_, predicted = torch.max(output, 1)
	print predicted
	print testdata['label'][:,0]
	total += testdata['label'][:,0].size(0)
	correct += (predicted == testdata['label'][:,0]).sum().item()
	predictions.writerow(predicted.numpy())
	testlabels.writerow(testdata['label'][:,0].numpy())

print('Accuracy of the network: %f %%' % (100 * correct/total))

