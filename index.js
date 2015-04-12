"use strict"
var ndarray = require("ndarray")
var ops = require("ndarray-ops")
var pool = require("ndarray-scratch")
var cwise = require("cwise")
require('array.from')

// TODO: Linear pyramids (e.g. Gaussian and/or spline)
// TODO: Implement detail pyramids (and reconstructions from such pyramids).
// TODO: Be more careful about boundary conditions.

module.exports = computePyramid
module.exports.adjunction = {reduce: adjunctionReduce}
module.exports.SunMaragos = {reduce: SunMaragosReduce}

function computePyramid(img, scheme, maxlevel) {
  var imgShape = Array.from(img.shape)

  if (maxlevel === undefined) maxlevel = Infinity
  
  var imgs = [img], tempIn, tempOut
  for(var level=1; level<=maxlevel && Math.max.apply(null, img.shape)>1; level++) {
    tempIn = poolClone(img)
    tempOut = scheme.reduce(tempIn)
    img = ndarrayClone(tempOut)
    imgs.push(img)
    imgShape = Array.from(img.shape)
    pool.free(tempIn)
  }
  
  return imgs
}

////////////////////////////////////////
// Morphological pyramids

function adjunctionReduce(img) {
  var dims = img.shape.length
  var steps = [], los = [], his = []
  var e = img.step(2), o = img.lo(1).step(2)
  for(var d=0; d<dims; d++) {
    pairwiseMin(img.shape[d]%2 === 0 ? e : e.hi.apply(e, his.concat([e.shape[d] - 1])), o)
    steps.push(1)
    los.push(0)
    his.push(e.shape[d])
    o = e.lo.apply(e, los.concat([1]))
    o = o.step.apply(o, steps.concat([2]))
    e = e.step.apply(e, steps.concat([2]))
  }
  return e
}

// This function assumes img was allocated from pool.
function SunMaragosReduce(img) {
  var dims = img.shape.length, steps, picks, los, his, e, o
  // First erode the image
  picks = []
  los = []
  his = []
  var img2 = pool.malloc(img.shape, img.dtype), tmpimg
  for(var d=0; d<dims; d++) {
    ops.assign(img2.pick.apply(img2, picks.concat([0])),
               img.pick.apply(img, picks.concat([0])))
    erode(img2.lo.apply(img2, los.concat([1])),
          img.lo.apply(img, los.concat([1])),
          img.hi.apply(img, his.concat([img.shape[d] - 1])))
    picks.push(null)
    los.push(0)
    his.push(img.shape[d])
    tmpimg = img
    img = img2
    img2 = tmpimg
  }
  pool.free(img2) // Could be the original argument, but we assume it also came from pool
  // Then dilate and subsample
  steps = [], los = [], his = []
  e = img.step(2), o = img.lo(1).step(2)
  for(var d=0; d<dims; d++) {
    pairwiseMax(img.shape[d]%2 === 0 ? e : e.hi.apply(e, his.concat([e.shape[d] - 1])), o)
    steps.push(1)
    los.push(0)
    his.push(e.shape[d])
    o = e.lo.apply(e, los.concat([1]))
    o = o.step.apply(o, steps.concat([2]))
    e = e.step.apply(e, steps.concat([2]))
  }
  return e
}

////////////////////////////////////////
// Helpers

var pairwiseMin = cwise({
  args: ["array", "array"],
  body: function(e, o) {
    e = Math.min(e, o)
  }
})

var pairwiseMax = cwise({
  args: ["array", "array"],
  body: function(e, o) {
    e = Math.max(e, o)
  }
})

var erode = cwise({
  args: ["array", "array", "array"],
  body: function(out, in1, in2) {
    out = Math.min(in1, in2)
  }
})

////////////////////////////////////////
// Helpers (that belong elsewhere)

// The clone function from ndarray-scratch copies the ENTIRE backing store, that is not the intended behaviour here.
function poolClone(arr) {
  var newArr = pool.malloc(arr.shape, arr.dtype)
  ops.assign(newArr, arr)
  return newArr
}

function ndarrayClone(arr) {
  var newArr = ndarrayFromDtype(arr.shape, arr.dtype)
  ops.assign(newArr, arr)
  return newArr
}

function ndarrayFromDtype(shape, dtype) {
  var size = shape.reduce(function(a,c){return a*c})
  var data
  switch(dtype) {
  case "int8":
    data = new Int8Array(size)
    break
  case "int16":
    data = new Int16Array(size)
    break
  case "int32":
    data = new Int32Array(size)
    break
  case "uint8":
    data = new Uint8Array(size)
    break
  case "uint16":
    data = new Uint16Array(size)
    break
  case "uint32":
    data = new Uint32Array(size)
    break
  case "float32":
    data = new Float32Array(size)
    break
  case "float64":
    data = new Float64Array(size)
    break
  case "array":
    data = new Array(size)
    break
  case "uint8_clamped":
    data = new Uint8ArrayClamped(size)
    break
  case "buffer":
    data = new Buffer(size)
    break
  default:
    throw new Error("Unrecognized data type.")
  }
  return ndarray(data, shape)
}
