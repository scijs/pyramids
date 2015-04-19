"use strict"
var assert = require("assert")
var ndarray = require("ndarray")
var ops = require("ndarray-ops")
var pool = require("ndarray-scratch")
var cwise = require("cwise")
var equals = require('array-equal')
require('array.from')

// TODO: Linear pyramids (e.g. Gaussian and/or spline)
// TODO: Implement detail pyramids (and reconstructions from such pyramids).
// TODO: Be more careful about boundary conditions.

module.exports = computePyramid
module.exports.detail = computeDetailPyramid
module.exports.reconstruct = reconstruct
module.exports.adjunction = {reduce: adjunctionReduce, expand: dilationExpand}
module.exports.SunMaragos = {reduce: SunMaragosReduce, expand: dilationExpand}

function computePyramid(img, scheme, maxlevel) {
  var imgShape = Array.from(img.shape)

  if (maxlevel === undefined) maxlevel = Infinity
  
  var imgs = [img], tempIn, tempOut
  for(var level=1; level<=maxlevel && Math.max.apply(null, img.shape)>1; level++) {
    tempIn = poolClone(img)
    tempOut = scheme.reduce(tempIn) // Reduce can be considered to return a view of tempIn (or a different array allocated from the pool).
    img = ndarrayClone(tempOut)
    imgs.push(img)
    imgShape = Array.from(img.shape)
    pool.free(tempOut) // We want to free tempOut, since reduce may have returned a different array from the pool and freed tempIn itself.
  }
  
  return imgs
}

function computeDetailPyramid(img, scheme, maxlevel) {
  var imgShape = Array.from(img.shape)
  var img = ndarrayClone(img) // Copy argument since we are going to overwrite it.

  if (maxlevel === undefined) maxlevel = Infinity
  
  var imgs = [], tempIn, tempOut, tempEx, newimg
  for(var level=1; level<=maxlevel && Math.max.apply(null, img.shape)>1; level++) {
    // Reduce
    tempIn = poolClone(img)
    tempOut = scheme.reduce(tempIn)
    newimg = ndarrayClone(tempOut)
    pool.free(tempOut)
    
    // Compute detail
    tempEx = pool.malloc(img.shape, img.dtype)
    scheme.expand(tempEx, newimg)
    ops.subeq(img, tempEx)
    pool.free(tempEx)

    // Store results and reset img(Shape)
    imgs.push(img)
    img = newimg
    newimg = undefined
    imgShape = Array.from(img.shape)
  }
  
  imgs.push(img) // Push the last level (not a detail level)
  
  return imgs
}

function reconstruct(detailPyramid, scheme) {
  var temp
  for(var level=detailPyramid.length-1; level-->0;) {
    temp = poolClone(detailPyramid[level])
    scheme.expand(temp, detailPyramid[level+1])
    ops.addeq(detailPyramid[level], temp)
  }
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

function dilationExpand(target, source) {
  var dims = source.shape.length
  var steps = [], los = []
  for(var d=0; d<dims; d++) {
    steps.push(2)
    los.push(0)
  }
  var e = target.step.apply(target, steps), o
  assert(equals(e.shape, source.shape))
  ops.assign(e, source)
  for(var d=0; d<dims; d++) {
    los[d] = 1
    o = target.lo.apply(target, los)
    o = o.step.apply(o, steps)
    dilateUpsample(e.hi.apply(e, o.shape), o)
    steps[d] = 1
    los[d] = 0
    e = target.step.apply(target, steps)
  }
  return target
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

var dilateUpsample = cwise({
  args: ["array", "array"],
  body: function(e, o) {
    o = e // This "dilation" uses a SE of length, with the assumption that odd values are (initially) zero.
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
