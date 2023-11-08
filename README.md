# bubbles
Bubble System for neural networks

Trainable torch model class that can hold a variable number of different neural networks inside "bubbles", which are connected to each other.  
Each tick, the held networks receive inputs from each other, process it and store it for the next tick.  
The model will return an output after a set amount of ticks.  

connection_size = 32
connections = 4 

```
                    Bubble_0    Bubble_1    Bubble_2    Bubble_3  
output total:       128         128         128         128  
output to Bubble_0: 32          32          32          32  
output to Bubble_1: 32          32          32          32  
output to Bubble_2: 32          32          32          32  
output to Bubble_3: 32          32          32          32  



Bubble_0 -- 32 -->\                                  / -- 32 -->  (one of the inputs of Bubble_0 next tick)
Bubble_1 -- 32 --> | -- 128 --> Bubble_0 -- 128 --> |  -- 32 -->  (one of the inputs of Bubble_1 next tick)
Bubble_2 -- 32 --> |                                |  -- 32 -->  (one of the inputs of Bubble_2 next tick)
Bubble_3 -- 32 -->/                                  \ -- 32 -->  (one of the inputs of Bubble_3 next tick)
```


##### Advantages:  
* might have better learning capabilites in complex / very diverse tasks  
* can load pretrained models in bubbles  
* more bubbles can be added after training  
* enables use of different architectures in a single model (e.g. Transformers, RNNs, custom models, etc.)  

##### Disadvantages:  
* inefficient in simple tasks
* for now inputs and outputs have to be the same size everywhere

### how to use
example:  
you can use any other model instead of bubbles.BaseModel, as long as the input and output size of that model are as big as connection_size
```
    model_base = bubbles.BaseModel(model_size = 128, model_layers = 3, connections = 4, connection_size = 32) # model input/output = connections * connection_size for now
    model1 = bubbles.BaseModel(model_size = 128, model_layers = 3, connections = 4, connection_size = 32)
    model2 = bubbles.BaseModel(model_size = 128, model_layers = 3, connections = 4, connection_size = 32)
    model3 = bubbles.BaseModel(model_size = 128, model_layers = 3, connections = 4, connection_size = 32)

    bubbles_array = [model1, model2, model3]


    model = bubbles.System(base_bubble = model_base, 
                   bubbles = bubbles_array,
                   ticks_to_run = 3, # 0 ticks outputs vector filled with 0s, 1 tick only uses base bubble. other bubbles only get used after the first tick. 
                   connection_size=32, # total connection size (single connection size = connection_size / num_connections)
                   output_size = dictionary_max_size, # input size is equal to base_bubble input size for now
                   ).to(device) # optionally move the model to cuda, recommended
```

##### future updates:  
* easier to use  
* better documentation  
* more features
* make base bubble into input_bubble (only used once before the first tick) and add output_bubble (only used once after last tick)

##### known problems:
* when using cpu instead of gpu (cuda): RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
