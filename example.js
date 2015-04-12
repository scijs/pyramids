"use strict"
var pyramid = require("./index.js")
var ndarray = require("ndarray")

console.log(pyramid(ndarray(new Int32Array([1, 2, 3, 4, 5, 6]), [6]), pyramid.adjunction))

console.log(pyramid(ndarray(new Int32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), [4,4]), pyramid.adjunction))

console.log(pyramid(ndarray(new Int32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5]), pyramid.adjunction))

console.log(pyramid(ndarray(new Int32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5]), pyramid.SunMaragos))
