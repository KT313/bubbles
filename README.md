# bubbles
Bubble System for neural networks

Trainable torch model class that can hold a variable number of different neural networks inside "bubbles", which are connected to each other.  
Each tick, the held networks receive inputs from each other, process it and store it for the next tick.  
The model will return an output after a set amount of ticks.  

##### Advantages:  
* inefficient in simple tasks  

##### Disadvantages:  
* might have better learning capabilites in complex / very diverse tasks
* can load pretrained models in bubbles
* more bubbles can be added after training
* enables use of different architectures in a single model (e.g. Transformers, RNNs, custom models, etc.)

##### future updates:  
* easier to use
* better documentation
* more features
