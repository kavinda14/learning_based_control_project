import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time


# Dataset
# def datasetGenerator(limit, partial_info_binary_matrices_pickle, path_matricies_pickle, final_actions_binary_matrices_pickle, final_scores_pickle): 
#     data = list()

#     for i in range(limit):

#         partial_info_pickle_in = open(partial_info_binary_matrices_pickle, "rb")
#         current_partial_info = pickle.load(partial_info_pickle_in)
        
#         path_matrices_pickle_in = open(path_matricies_pickle, "rb")
#         current_path_matrix = pickle.load(path_matrices_pickle_in)
#         action_pickle_in = open(final_actions_binary_matrices_pickle, "rb")
#         current_action = pickle.load(action_pickle_in)
#         scores_pickle_in = open(final_scores_pickle, "rb")
#         current_score = pickle.load(scores_pickle_in)
#         print(len(current_score))

#         for j in range(len(current_partial_info)):
#             image = list()
            
#             for partial_info in current_partial_info[j]:
#                 image.append(partial_info)

#             image.append(current_path_matrix[j])

#             for action in current_action[j]:
#                 image.append(action)
            
            
#             data.append([torch.IntTensor(image), current_score[j]])

#     return data

# Dataset
def datasetGenerator(partial_info_binary_matrices, path_matricies, final_actions_binary_matrices, final_scores): 
    data = list()

    for i in range(len(partial_info_binary_matrices)):
        image = list()

        for partial_info in partial_info_binary_matrices[i]:
            image.append(partial_info)

        image.append(path_matricies[i])

        for action in final_actions_binary_matrices[i]:
            image.append(action)
        
        
        data.append([torch.IntTensor(image), final_scores[i]])

    return data

# This was created for when using a planner with the network
def create_image(partial_info_binary_matrices, path_matricies, final_actions_binary_matrices):
    image = list()

    for i in range(len(partial_info_binary_matrices)):
        
        for partial_info in partial_info_binary_matrices[i]:
            image.append(partial_info)

        image.append(path_matricies)

        for action in final_actions_binary_matrices[i]:
            image.append(action)
        
    return torch.IntTensor(image)



class PlanningDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        # The label is converted to a float and in a list because the NN will complain otherwise.
        sample = self.data[idx][0], torch.Tensor([self.data[idx][1]]).float()

        return sample

# Dataloaders
def createDataLoaders(data):
    train_data = PlanningDataset(data)
    test_data = PlanningDataset(data)

    batch_size = 128

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
                                            
    return [trainloader, testloader]

# Neural Network architecture
def get_linear_layer_multiple(value):
    return (value - 5) + 1


class Net(nn.Module):
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds
        # input channels, output no. of features, kernel size
        self.conv1 = nn.Conv2d(7, 12, 5)
#         self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        # self.fc1 = nn.Linear(16 * 103 * 103, 120)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv1(x)))
        
#         x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
     #   x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def runNetwork(data, bounds):
    
    trainloader, testloader = createDataLoaders(data)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(bounds)
    # net = Net(bounds).to(device)

    # Loss + Optimizer
    criterion = nn.MSELoss()
    # SGD produced too low loss values and forums recommended Adam
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Run network
    start = time.time()
    loss_values = list()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # REMEMBER TO ADD float()
            outputs = net(inputs.float())
            
            # print("outputs: ", outputs)
            # print("labels: ", labels)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            # print(loss.item())
            if i % 100 == 99:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                loss_values.append(running_loss/100)
                running_loss = 0.0            
    
    end = time.time()
    time_taken = (end - start)/60
    print("Time taken: {:.3f}".format(time_taken))

    torch.save(net.state_dict(), "/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2")
    print('Finished Training')

    loss_values.append(running_loss)
    plt.plot(loss_values)
    plt.title("Loss Values, time taken: {:.4f}".format(time_taken))
    plt.show()