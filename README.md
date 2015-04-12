# Pyramids

Image pyramids give a (slightly) overcomplete multiscale representation of an image, based on a reduce operator that reduces the size of an image by a factor of two (note though that conventionally, level 0 is the original scale, and higher levels correspond to smaller sizes). Example usage:

```javascript
var pyramid = require("pyramids")
var ndarray = require("ndarray")

var p = pyramid(ndarray(new Int32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5]), pyramid.adjunction)
```

The above computes a pyramid using erosion (local minima, in this case within each 2-by-2 block), of the following "image" (array):

<table>
<tr><td align="right">11</td><td align="right">10</td><td align="right">6</td><td align="right">12</td><td align="right">3</td></tr>
<tr><td align="right">2</td><td align="right">15</td><td align="right">9</td><td align="right">4</td><td align="right">5</td></tr>
<tr><td align="right">14</td><td align="right">7</td><td align="right">13</td><td align="right">8</td><td align="right">1</td></tr>
</table>

The result is a list of ndarrays, with the first array (corresponding to level 0) being the original ndarray object. The array at level 1 is given by:

<table>
<tr><td align="right">2</td><td align="right">4</td><td align="right">3</td></tr>
<tr><td align="right">7</td><td align="right">8</td><td align="right">1</td></tr>
</table>

Note that although typically an expand operator is also given, allowing one to construct a detail pyramid (containing just the difference between each level and the reconstruction from the next higher level) and reconstructing an image from such a pyramid, this has not yet been implemented in this module.

### `require("pyramids")(array, scheme[, maxlevel])`

Returns a list containing `array`, followed by `maxlevel` (higher) levels of a pyramid based on the reduce operator of `scheme` (each an ndarray with the same type of data storage as `array`). If `maxlevel` is unspecified, it returns as many levels are necessary to end up with an array containing just one element.

There is no restriction on the dimensionality of `array`, but a typed array must be used for storing the data in `array` (otherwise the pool allocator used internally will fail).

## Morphological pyramids.

Non-linear pyramids based on reduce and expand operators that satisfy the so-called pyramid condition: first expanding an image and then reducing it recovers the original image.

### `pyramid.adjunction`

The adjunction pyramid has a reduce operator based on erosion, as described in

> Nonlinear multiresolution signal decomposition schemes &mdash; Part I: Morphological pyramids IEEE Transactions on Image Processing, Vol. 9, No. 11. (November 2000), pp. 1862-1876, doi:[10.1109/83.877209](http://dx.doi.org/10.1109/83.877209) by John Goutsias, Henk J. A. M. Heijmans.

The structuring element used here is a (flat) 2-by-2 square in 2D, and in general a hypercube with sides of length 2.

### `pyramid.SunMaragos`

The SunMaragos pyramid has an opening (rather than an erosion) as reduce operator. The same structuring element is used as in the adjunction pyramid, based on the results in

> A New Class of Morphological Pyramids for Multiresolution Image Analysis In Geometry, Morphology, and Computational Imaging, Vol. 2616 (2003), pp. 165-175, doi:[10.1007/3-540-36586-9_11](http://dx.doi.org/10.1007/3-540-36586-9_11) by Jos B. T. M. Roerdink edited by Tetsuo Asano, Reinhard Klette, Chrisitan Ronse.

This pyramid tends to preserve more of the image than the adjunction pyramid. Note that (according to Goutsias and Heijmans, 2000) Sun and Maragos originally used a different structuring element (of length 3).
