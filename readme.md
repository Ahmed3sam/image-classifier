# deep learning image classifier

## dataset
the dataset given was 495 train-data of *9 classes*

### problems
1. the dataset is very small
2. the labels in outlayer in the model arranged *alphabetically* (including upper and lower case alphabet)
3. there is no folder for validation data 

### solutions
1. using augmentation to make the model train on many scenarios
2. make the labels sorted alphabetically before testing
3. take 10% of the train data for validation (only small set because it is already small)

## model
the model is CNN has 4 layers + final DNN layer 
it has drop out to prevent overfitting 

##results
the results was not bad but also not good
training accuracy: 83.4% (increasing gradually)
validation accuracy: 67.92% (increasing in zigzag shapes due to the small dataset)
![alt text](https://github.com/Ahmed3sam/image-classifier/image/figure.PNG)


* I think that test accuracy will be about 55 - 60 %


## solutions
We can use Transfer learning in pre trained model like inception or VGG and retrain it on our dataset
it will get more accuracy 
