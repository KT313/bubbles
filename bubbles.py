import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from .settings import *

class BaseModel(nn.Module):
    def __init__(self, connections, connection_size, model_size, model_layers):
        super(BaseModel, self).__init__()
        # print(f"basemodel: {connection_size} {connections} {connection_size*connections}")
        self.layer_input = nn.Linear(connection_size*connections, model_size)
        self.layers_middle = nn.ModuleList(nn.Linear(model_size, model_size) for _ in range(model_layers))
        self.layer_output = nn.Linear(model_size, connection_size*connections) #nn.ModuleList(nn.Linear(model_size,connection_size) for _ in range(connections))
        self.relu = nn.ReLU()
        self.connections = connections
        self.connection_size = connection_size
        # print(f"input_layer: {self.layer_input.weight.shape}, middle layer: {self.layers_middle[0].weight.shape}")

    def forward(self, x):
        # print(f"shape of BaseModel input: {x.shape}")
        # Reshape the input while keeping the batch size the same
        batch_size = x.size(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.view(batch_size, -1).to(device)
        # print(f"reshaped BaseModel input: {x.shape}, layer shape: {self.layer_input.weight.shape}")
        x = self.relu(self.layer_input(x))
        for layer_middle in self.layers_middle:
            x = self.relu(layer_middle(x))
        # Instead of multiple output layers, use one layer that outputs all connections at once
        # Then, reshape to get the desired output shape
        x = self.layer_output(x)
        outputs = x.view(batch_size, self.connections, self.connection_size)
        # print(f"shape of BaseModel outputs: {outputs.shape}")
        return outputs

class InputBaseModel(nn.Module):
    def __init__(self, connections, connection_size, model_size, model_layers, input_size = 32):
        super(OutputBaseModel, self).__init__()
        # print(f"basemodel: {connection_size}")
        self.layer_input = nn.Linear(input_size, model_size)
        self.layers_middle = nn.ModuleList(nn.Linear(model_size, model_size) for _ in range(model_layers))
        self.layer_output = nn.Linear(model_size, output_size) #nn.ModuleList(nn.Linear(model_size,connection_size) for _ in range(connections))
        self.relu = nn.ReLU()
        self.connections = connections
        self.connection_size = connection_size
        self.output_size = output_size

    def forward(self, x):
        # print(f"shape of BaseModel input: {x.shape}")
        # Reshape the input while keeping the batch size the same
        batch_size = x.size(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.view(batch_size, -1).to(device)
        # print(f"reshaped BaseModel input: {x.shape}")
        x = self.relu(self.layer_input(x))
        for layer_middle in self.layers_middle:
            x = self.relu(layer_middle(x))
        # Instead of multiple output layers, use one layer that outputs all connections at once
        # Then, reshape to get the desired output shape
        x = self.layer_output(x)
        outputs = x.view(batch_size, self.output_size)
        # print(f"shape of BaseModel outputs: {outputs.shape}")
        return outputs


class OutputBaseModel(nn.Module):
    def __init__(self, connections, connection_size, model_size, model_layers, output_size):
        super(OutputBaseModel, self).__init__()
        # print(f"basemodel: {connection_size}")
        self.layer_input = nn.Linear(connection_size, model_size)
        self.layers_middle = nn.ModuleList(nn.Linear(model_size, model_size) for _ in range(model_layers))
        self.layer_output = nn.Linear(model_size, output_size) #nn.ModuleList(nn.Linear(model_size,connection_size) for _ in range(connections))
        self.relu = nn.ReLU()
        self.connections = connections
        self.connection_size = connection_size
        self.output_size = output_size

    def forward(self, x):
        # print(f"shape of BaseModel input: {x.shape}")
        # Reshape the input while keeping the batch size the same
        batch_size = x.size(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.view(batch_size, -1).to(device)
        # print(f"reshaped BaseModel input: {x.shape}, layer shape: {self.layer_input.weight.shape}")
        x = self.relu(self.layer_input(x))
        for layer_middle in self.layers_middle:
            x = self.relu(layer_middle(x))
        # Instead of multiple output layers, use one layer that outputs all connections at once
        # Then, reshape to get the desired output shape
        x = self.layer_output(x)
        outputs = x.view(batch_size, self.output_size)
        # print(f"shape of BaseModel outputs: {outputs.shape}")
        return outputs








class Bubble(nn.Module):
    def __init__(self, model, num_connections, connection_size,id, is_base_bubble=False):
        super(Bubble, self).__init__()
        # print(f"bubble: {connection_size}")
        # Initialize the neural network based on the specified architecture
        self.id = id
        self.num_connections = num_connections
        self.neural_network = model
        self.is_base_bubble = is_base_bubble
        self.inputs = [[0 for _ in range(connection_size)] for _ in range(num_connections)]
        self.outputs = [[0 for _ in range(connection_size)] for _ in range(num_connections)] # stores nn result so it can be passed on next tick
        if self.is_base_bubble:
            self.outside_input = None
            self.outside_output = None
        pass

    def do_tick(self, inputs):
        # Process inputs and return outputs
        # # print(inputs.shape)
        self.inputs = inputs.squeeze(0)
        # # print(self.inputs.shape)
        outputs = self.neural_network(self.inputs)
        # print(f"doing tick for bubble {self.id}\n\tInput: {self.inputs.shape}, Output: {outputs.shape}")
        return outputs
        pass

    def get_output(self):
        # Return the output, but don't process new inputs yet
        pass

class System(nn.Module):
    def __init__(self, bubble_count = None, ticks_to_run = 3, connection_size = 32, base_bubble = None, bubbles = [], outside_input = np.zeros(64).astype(float), output_size = 32, input_dtype = torch.float):
        super(System, self).__init__()
        # print(f"system: {connection_size}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_bubbles = 1 + len(bubbles) # 1 for base_bubble
        self.num_connections = self.num_bubbles
        if base_bubble == None:
            print("error, base bubble missing")
            # self.base_bubble = Bubble(is_base_bubble=True, num_connections = bubble_count, connection_size = connection_size, model_size = model_size, model_layers = model_layers, id = 0)
        else:
            self.base_bubble = Bubble(model = base_bubble, id = 0, connection_size = connection_size, num_connections = self.num_connections, is_base_bubble = True)

        if bubbles == []:
            # for i in range(bubble_count-1):
            #     bubbles.append(Bubble(num_connections = bubble_count, connection_size = connection_size, model_size = model_size, model_layers = model_layers, id = i+1))
            # self.bubbles = nn.ModuleList(bubbles)
            print("error, bubbles missing")
        else:
            bubble_merker = []
            counter = 1
            for part in bubbles:
                bubble_merker.append(Bubble(model = part, id = counter, connection_size = connection_size, num_connections = self.num_connections))
                counter += 1
            self.bubbles = nn.ModuleList(bubble_merker)
        self.ticks = 0
        self.connection_size = connection_size
        self.ticks_to_run = ticks_to_run
        self.outside_input = outside_input
        self.output_size = output_size
        self.outputmodel = OutputBaseModel(connections = 1, connection_size = connection_size, model_size = 128, model_layers = 2, output_size = self.output_size).to(self.device)
        self.connection_buffer = torch.zeros((batch_size, self.num_bubbles, self.num_bubbles, self.connection_size), requires_grad=False) # saves outputs of bubbles and provides them as inputs next tick. [0, 1] -> output from bubble 0, to be used for bubble 1

    def tick(self, outside_input):
        # Logic for one tick of the system
        # print(f"----------\nDoing tick {self.ticks}\n")
        # print(f"connection_buffer shape at start of tick: {self.connection_buffer.shape}")

        # # print(f"outside input shape: {outside_input.shape}\ncontent:\n{outside_input}")
        if self.ticks == 0:
            # Only the base bubble processes input
            # print(f"outside input shape \nbefore unsqueezing: {outside_input.shape}\nafter unsqueezing: {outside_input.unsqueeze(1).shape}\ntensor to cat: {torch.zeros((outside_input.shape[0] ,self.num_bubbles - 1, outside_input.size(0))).shape}")
            filler_tensor = torch.zeros((outside_input.shape[0], self.num_bubbles - 1, self.connection_size))
            # print(f"filler tensor shape: {filler_tensor.shape}\ninput tensor shape: {outside_input.shape}\ninput tensor unseqeezed shape: {outside_input.unsqueeze(1).shape}")
            input_tensor = torch.cat([outside_input.unsqueeze(1), filler_tensor.to(self.device)], dim=1)
            # print(f"shape after filling: {input_tensor.shape}")
            # # print(f"self.outside_input {self.outside_input.shape}, resulting input {input.shape}")
            output = self.base_bubble.do_tick(inputs = input_tensor)
            # print(f"output shape from bubble: {output.shape}\nshape connection buffer: {self.connection_buffer.shape}")
            for i in range(output.shape[0]): #for each batch part
                # print(f"whole buffer: {self.connection_buffer.shape}\nbuffer to insert in: {self.connection_buffer[i, self.base_bubble.id].shape}, to insert: {output[i].shape}")
                self.connection_buffer[i, self.base_bubble.id].copy_(output[i])
            pass
        else:
            # base bubble then other bubbles
            tensor_output = self.base_bubble.do_tick(inputs=self.connection_buffer[:,0])
            # print(f"tensor output shape: {tensor_output.shape}")
            for i in range(tensor_output.shape[0]): #for each batch part
                self.connection_buffer[i, self.base_bubble.id].copy_(tensor_output[i])
            
            # now other bubbles
            for i in range(self.num_bubbles-1):
                tensor_output = self.bubbles[i].do_tick(inputs=self.connection_buffer[:,i+1])
                # outputs.append(tensor_output)
                for batch_nr in range(tensor_output.shape[0]): #for each batch part
                    self.connection_buffer[batch_nr, self.bubbles[i].id].copy_(tensor_output[batch_nr])
            pass
        self.ticks += 1

    def forward(self, x):
        start_run_time = time.time()
        tick_time = []
        wrong_batch_size = False
        # print(x.shape)
        # if the outside input is smaller than total connection size, fill missing values with 0
        if x.shape[0] < batch_size:
            wrong_batch_size = True
            merker_batch_size = x.shape[0]
            c = torch.zeros(batch_size, x.shape[1])
            c[:x.shape[0]] = x
            x = c
        a = torch.zeros(batch_size, self.connection_size).to(self.device)
        # print(a.shape)
        a[:, :x.shape[-1]] = x
        x = a
        # print(x.shape)
      # print(x.shape)
        # Run the system for a certain number of ticks
        for _ in range(self.ticks_to_run):
            start_tick_time = time.time()
            self.tick(outside_input = x)
            tick_time.append(time.time()-start_tick_time)
        output = self.connection_buffer[:,0,0]
        output = self.outputmodel(output)

        if wrong_batch_size:
            # print(output.shape)
            output = output[:merker_batch_size]
            # print(output.shape)
        # reset
        self.ticks = 0
        # print(f"self.connection_buffer shpae: {self.connection_buffer.shape}")
        self.connection_buffer = torch.zeros((batch_size, self.num_bubbles,self.num_bubbles,self.connection_size), requires_grad=False)
        # print(f"\n\n\nsystem output shape:\n{output.shape}\n")
        # print(f"avg tick time: {(sum(tick_time)/self.ticks_to_run)*1000:.2f}ms")
        # print(f"total run time: {(time.time()-start_run_time)*1000:.2f}ms")
        return output
