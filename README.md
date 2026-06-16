what did i learn:
- for backprop, each layer's gradient can be seen as upstream (because it is coming form the the layer ahead) and this is multiplied by the current layer's derivative to see how much the current layer's neurons are affecting the final cost

- for backprop, the error signal (cost) is what flows back. The activations are cached to be reused to compute the updated weights in forward-prop as backprop needs activation_prev to compute dC/dw and without caching, there is no way of recovering them.

- data imbalance is the first thing that should always be handled if possible.
