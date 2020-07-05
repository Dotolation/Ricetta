# Automatic Cookpad Recipe Title Generator

Given a Cookpad recipe written in Japanese, the program will automatically predict an appropriate recipe title. 
RNN (Encoder-Decoder Model) with Bahdanau additive attention was used to create the predictive model. 
Python 3 and TensorFlow 2.0 was used to create the deep neural network. 

# How to Run
The test and training data (```test_BD, test_TI, train_BD, train_TI ```) are already preprocessed, so no action is required for pre-processing step.
To train a model and test the model, simply execute ```RNN_Toy.py```

#Overview

## Preprocessing
This phase cleans up raw recipe data and performs word segementation. 

### 1. Cleaning
   This task is performed by ```preprocessing.py```. By calling ```preprocessing``` function on the training/testing txt file, it will remove all the ascii arts deemed unnecessary as they tend to harm the performance. 
   Other informations like recipe IDs or Dats were also replaced by tokens '<ID>' and '<Date>' to reduce the data complexity.
   In Japanese, there are two versions of same characters: half-width(5) and full-width(５). 
   Since this also adds unneeded complexity, they were all mapped to the half-width characters. 

### 2. Word Segmentation
   Word segmentation is tricky in Japanese as words are not separated by a space. So I have used _ _KyTea_ _, a Japanese Text Analysis Toolkit developed in Kyoto University.
   (Creators are Graham Neubig, Yosuke Nakata, and Shinsuke Mori).
   For more information about KyTea, visit [its official page](http://www.phontron.com/kytea/).  
   The resulting file was a recipe with a space between words; this allows tokenizer to build a vocabulary list. 
   
### 3. Title/Body Seperation
   Finally, using ```preprocessing.py``` again, I have saved title and recipe body in seperate files. 
   This part was easy as title/body boundaries were clearly marked in the raw data. 
   
## Training 
In this part, the details about the encoder-decoder model is explained. 
Please refer to ```RNN_toy.py``` for the code. 

### 1. Tokenizing $ Generating Word Vectors
    Tokenizer was fed with the title (```train_TI```) and the body (```train_BD```) of the training data.
    Note: Excessively long recipes were excluded in the training data due to the speed issue. 
    
### 2. Encoder & Decoder
    Encoder has embedding layer and the main layer. Embedding dimension was set to 512. 
    Decoder has an additional dense layer compareed to the encoder. 
    GRU was used instead of LSTM due to the speed issue. 

### 3. Attention Mechanism
    Additive attention with two attention weight matrix and one attention vector was used. 
    This is also known as Bahdanau Attention. 

### 4. Batch size & Epochs
    Batch size was set to 32. Epoch was set to 40.
    
### 5. Saving the Model 
    The model will be saved in the directory './training_checkpoints", which will be generated when the code is run.
    
