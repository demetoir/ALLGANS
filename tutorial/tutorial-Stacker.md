# Tutorial Stacker

## Detail
help easily make graph model by stacking layer and naming for tensorflow

## Usage

```python
# start with input tensor tensorflow.Placeholder of tensorflow.Variable
layer = layerModel(input_)

# using add_layer function to add new layer
# feed function at util.tensor_ops
layer.add_layer(conv2d, 64, CONV_FILTER_5522)
layer.add_layer(bn)
layer.add_layer(lrelu)

# or just use shortcut function
layer.conv2d(128,CONV_FILTER_5522)
layer.bn()
layer.lrelu()

# or simply call block function
layer.conv2d_block(256, CONV_FILTER_5522, lrelu)

# access last_layer of model
output = layer.last_layer
```

