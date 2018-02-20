# Tutorial LayerModel

## Detail
easier to make ML model

## Usage step

1.
```python
layer = layerModel(input_)
layer.add_layer(conv2d, 64, CONV_FILTER_5522)
layer.add_layer(bn)
layer.add_layer(lrelu)

layer.add_layer(conv2d, 128, CONV_FILTER_5522)
layer.add_layer(bn)
layer.add_layer(lrelu)

layer.add_layer(conv2d, 256, CONV_FILTER_5522)
layer.add_layer(bn)
layer.add_layer(lrelu)
output = layer.last_layer
```

