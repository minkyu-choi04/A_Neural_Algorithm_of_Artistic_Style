# A Neural Algoritm of Artistic Style
Pytorch reproduction of the paper ["A Neural Algoritm of Artistic Style"](https://arxiv.org/pdf/1508.06576.pdf "Paper Link"). Some improvements are made by adopting ideas from ["A Learned Representation for Artistic Style"](https://arxiv.org/pdf/1610.07629.pdf)
1. Zero-padding is replaced with mirror-padding. 
2. Transposed convolution is replaced with up-sampling and covolution. 

## Dependencies
```
python 3.6.5
pytorch 0.4.1.post2
```

## Usage
```
python main.py
```
You can run your own experiment by giving parameters manually. 

## Results
Original content image (Cape Manzamo, Okinawa, Japan): 

<img src="https://github.com/minkyu-choi04/A_Neural_Algorithm_of_Artistic_Style/blob/master/content.jpg" alt="Original Content Image" width="500"/>



Style image:

<img src="https://github.com/minkyu-choi04/A_Neural_Algorithm_of_Artistic_Style/blob/master/style.jpg" alt="Style Image" width="500"/>



Resulting image (after 49 iterations):

<img src="https://github.com/minkyu-choi04/A_Neural_Algorithm_of_Artistic_Style/blob/master/sample_output/outout49.jpg" alt="Resulting Image" width="500"/>






