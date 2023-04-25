import torch
from torch.utils.data import DataLoader
from dataset import CubeDataset
from model import Img2PcdModel
from loss import CDLoss, HDLoss
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():

    # TODO: Design the main function, including data preparation, training and evaluation processes.

    # Environment:
    # device: torch.device

    # Directories:
    # cube_data_path: str, cube dataset root directory
    # output_dir: str, result directory

    # Training hyper-parameters:
    # batch_size: int
    # epoch: int
    # learning_rate: float

    # Data lists:
    # training_cube_list: list
    # test_cube_list: list
    # view_idx_list: list

    # Preperation of datasets and dataloaders:
    # Example:
    #     training_dataset = CubeDataset(cube_data_path, training_cube_list, view_idx_list, device=device)
    #     test_dataset = CubeDataset(cube_data_path, test_cube_list, view_idx_list, device=device)
    #     training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    #     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Network:
    # Example:
    #     model = Img2PcdModel(device=device)

    # Loss:
    # Example:
    #     loss_fn = CDLoss()

    # Optimizer:
    # Example:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training process:
    # Example:
    #     for epoch_idx in range(epoch):
    #         model.train()
    #         for batch_idx, (data_img, data_pcd) in enumerate(training_dataloader):
    #             # forward
    #             pred = model(data_img)

    #             # compute loss
    #             loss = loss_fn(pred, data_pcd)

    #             # backward
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
        
    # Final evaluation process:
    # Example:
    #     model.eval()
    #     for batch_idx, (data_img, data_pcd, data_r) in enumerate(test_dataloader):
    #         # forward
    #         pred = model(data_img, data_r)
    #         # compute loss
    #         loss = loss_fn(pred, data_pcd)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    cube_data_path = '../Problem2/cube_dataset/noisy/'
    output_dir = '/output/'
    
    training_cube_list = list(range(80))
    test_cube_list = list(range(80, 100))
    view_idx_list = list(range(15))
    
    batch_size = 64
    epoch = 50
    learning_rate = 0.01
    
    
    training_dataset = CubeDataset(cube_data_path, training_cube_list, view_idx_list, device=device)
    test_dataset = CubeDataset(cube_data_path, test_cube_list, view_idx_list, device=device)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = Img2PcdModel(device=device)
    loss_fn = CDLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch_idx in range(epoch):
        model.train()
        for batch_idx, (data_img, data_pcd) in enumerate(training_dataloader):
            # print(batch_idx, data_img.shape, data_pcd.shape)
            pred = model(data_img)
            
            loss = loss_fn(pred, data_pcd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                      .format(epoch_idx + 1, epoch, batch_idx + 1, len(training_dataloader), loss.item()))
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                pred = pred.view(-1,3).cpu().detach().numpy().squeeze()
                data_pcd = data_pcd.view(-1, 3).cpu().detach().numpy().squeeze()
                ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', marker='o', s=0.1)
                ax.scatter(data_pcd[:, 0], data_pcd[:, 1], data_pcd[:, 2], c='b', marker='o', s=0.1)
                
                plt.savefig(f'output/train/{epoch_idx}_{batch_idx}.png')

    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (data_img, data_pcd) in enumerate(test_dataloader):
            pred = model(data_img)

            loss = loss_fn(pred, data_pcd)
            total_loss += loss.item()
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            pred = pred.view(-1,3).cpu().numpy().squeeze()
            data_pcd = data_pcd.view(-1, 3).cpu().numpy().squeeze()
            ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', marker='o', s=0.1)
            ax.scatter(data_pcd[:, 0], data_pcd[:, 1], data_pcd[:, 2], c='b', marker='o', s=0.1)
            
            plt.savefig(f'output/test/{batch_idx}.png')
            

        avg_loss = total_loss / len(test_dataloader)
        print('Average Test Loss: {:.4f}'.format(avg_loss))
        


if __name__ == "__main__":
    main()
