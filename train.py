from model import *
from torch_helper import *

import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import numpy.random as npr
import scipy.misc
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib


def train(args, x_train, y_train, x_test, y_test, colours, model_mode=None, model=None):
    # Set the maximum number of threads to prevent crash in Teaching Labs
    #####################################################################################
    # TODO: Implement this function to train model and consider the below items         #
    # 0. read the utils file and use 'process' and 'get_rgb_cat' to get x and y for     #
    #    test and train dataset                                                         #
    # 1. Create train and test data loaders with respect to some hyper-parameters       #
    # 2. Get an instance of your 'model_mode' based on 'model_mode==base' or            #
    #    'model_mode==U-Net'.                                                           #
    # 3. Define an appropriate loss function (cross entropy loss)                       #
    # 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
    # 5. Implement the main loop function with n_epochs iterations which the learning   #
    #    and evaluation process occurred there.                                         #
    # 6. Save the model weights                                                         #
    # Hint: Modify the predicted output form the model, to use loss function in step 3  #
    #####################################################################################
    """
    Train the model
    
    Args:
     model_mode: String
    Returns:
      model: trained model
    """
    torch.set_num_threads(5)

    
    np.random.seed(args.seed)

    
    save_dir = "outputs/" + args.experiment_name
    
    #Transforming Data
    num_colours = np.shape(colours)[0]
    print("Transforming data...")
     
    train_rgb, train_grey = process(x_train, y_train, downsize_input=args.downsize_input, category_id=args.category_id)
    train_rgb_cat = rgb2label(train_rgb, colours, args.batch_size)
    test_rgb, test_grey = process(x_test, y_test, downsize_input=args.downsize_input, category_id=args.category_id)
    test_rgb_cat = rgb2label(test_rgb, colours, args.batch_size)
    
       
    # LOAD THE MODEL
    num_in_channels = 1 if not args.downsize_input else 3
    if args.model == "CNN":
        MyModel = BaseModel(args.kernel, args.num_filters, num_colours, num_in_channels)
        print("CNN")
    elif args.model == "UNet":
        MyModel = CustomUNET(args.kernel, args.num_filters, num_colours, num_in_channels)
        print("UNet")
    elif args.model == "ResNet":
        MyModel = ResNet(args.kernel, args.num_filters, num_colours, num_in_channels)
        print("ResNet")
        
        
    # LOSS FUNCTION and Optimizer   
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(MyModel.parameters(), lr=args.learn_rate)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Beginning training ...")
    
    start = time.time()

    train_losses = []
    valid_losses = []
    valid_accs = []

    # Training loop
    for epoch in range(args.epochs):
        # Train the Model
        MyModel.train()  
        losses = []
        
        # Forward + Backward + Optimize
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb_cat,
                                               args.batch_size)):
            
            images, labels = get_torch_vars(xs, ys, False)

            
            optimizer.zero_grad()
            outputs = MyModel(images)

            
            loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colours=num_colours)
        
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())


       # Calculate and Print training loss for each epoch
        if args.plot:
            _, predicted = torch.max(outputs.data, 1, keepdim=True)
            plot(xs, ys, predicted.cpu().numpy(), colours,
                 save_dir+'/train_%d.png' % epoch, 
                 args.visualize, 
                 args.downsize_input)

        
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (
            epoch+1, args.epochs, avg_loss, time_elapsed))

        # Evaluate the model
        MyModel.eval()  
        val_loss, val_acc = run_validation_step(MyModel,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colours,
                                                save_dir+'/test_%d.png' % epoch,
                                                args.visualize,
                                                args.downsize_input)

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d' % (
            epoch+1, args.epochs, val_loss, val_acc, time_elapsed))

     # Calculate and Print (validation loss, validation accuracy) for each epoch
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir + "/training_curve.png")

    if args.checkpoint:
        print('Saving model...')
        torch.save(model.state_dict(), args.checkpoint)

    return MyModel
 