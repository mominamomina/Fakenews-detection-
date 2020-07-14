# train.py

from utils import *
from model import Transformer
from utils import Dataset
from model import *
from config import Config
from sklearn import metrics
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__=='__main__':
    
    config = Config()
    
    train_file = 'new 1.txt'
    print(train_file)
    #print("here")
    #print(type(train_file))
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = 'alice1.txt'
    
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = Transformer(config, len(dataset.vocab))
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################
    torch.save(model.state_dict(), 'model.pt')
    print("model saved")
    
    train_losses = []
    #val_losses =[]
    val_accuracies = []
    train_accuracies = []
    validation_losses = []
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy,train_accuracy, val_loss = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        #val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_losses)
        train_losses.append(train_loss)
    train_acc= evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)
    preci_acc= eva(model, dataset.test_iterator)
    recall_sco=eva1(model, dataset.test_iterator)
    f1_score=eva2(model, dataset.test_iterator)
    cnf_matrix=eva3(model, dataset.test_iterator)
    confusionmatrix=eva3(model, dataset.test_iterator)
    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    print ('Precision: {:.4f}'.format(preci_acc))
    print ('recall : {:.4f}' .format(recall_sco))
    print ('F1- score : {:.4f}' .format(f1_score))
    print (" *********** Confusion Matrix  **************")
    print(cnf_matrix)