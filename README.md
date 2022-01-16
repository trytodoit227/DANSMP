# Stock Movement Prediction Based on Bi-typed and Hybrid-relational Market Knowledge Graph via Dual Attention Networks

## Requirement Environment
- Python 3.6.13
- Pytorch 1.7.1
- Geometric 1.7.2

## Run
```sh
$ python main1.py
```
- Make sure that the GPU is used to reproduce our experiments.

## Data
The two datasets for SMP with their folder names are given below.
- CSI100E 
- CSI300E.

### Selected Stock
- The selected stocks in CSI100E and CSI300E can be found at ./raw_data/100E/stocks.txt and ./raw_data/100E/stocks.txt,respectively
 
### Transcational Data
- Due to The raw transcational data are about 1.1Gb and the limited space, a small part of the data is given. more data and preprocess code will be released soon.

- A part of transcational data can be found at ./raw_data/100E/historical price.xlsx and ./raw_data/300E/historical price.xlsx.
### Sentiment Indicators
- Due to the limited space and news data are about 2.3Gb, a small part of the data is given. more data and preprocess code will be released soon.  

- A part of financial news data can be found at ./raw_data/100E/financial news.xlsx and ./raw_data/300E/financial news.xlsx.

- The finance-oriented sentiment dictionary (found at ./raw_data/sentiment_dictionary) is used to extract sentiment from financial news.
### inter-class relations
- The inter-class relations data can be found at ./raw_data/100E/inter-class and ./raw_data/300E/inter-class. 


### intra-class Relations
- The intra-class relations can be found at ./raw_data/100E/intra-class and ./raw_data/300E/intra-class.

## Contact
duhmfcc@gmail.com
