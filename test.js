"use strict"
var test = require('tape')
var pyramid = require("./index.js")
var ops = require('ndarray-ops')
var ndarray = require('ndarray')
var cwise = require("cwise")

// The references (also) use fairly basic boundary conditions, when this is fixed in the main code, this should be fixed here as well.

var almostEqual = cwise({
  args: ["array", "array", "scalar", "scalar"],
  body: function(a, b, absoluteError, relativeError) {
    var d = Math.abs(a - b)
    if(d > absoluteError && d > relativeError * Math.min(Math.abs(a), Math.abs(b))) {
      return false
    }
  },
  post: function() {
    return true
  }
})

test("adjunction pyramid", function(t) {
  var data = ndarray(new Int32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5])
  var ref = [data, ndarray(new Int32Array([2,4,3,7,8,1]), [2,3]), ndarray(new Int32Array([2,1]), [1,2]), ndarray(new Int32Array([1]), [1,1])]

  var p = pyramid(data, pyramid.adjunction)

  t.equal(p.length, ref.length, "number of levels")
  for(var i=0; i<Math.min(p.length, ref.length); i++) {
    t.ok(ops.equals(p[i], ref[i]), "level " + i)
  }

  t.end()
})

test("adjunction detail pyramid", function(t) {
  var data = ndarray(new Int32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5])
  var ref = [ndarray(new Int32Array([9,8,2,8,0,0,13,5,0,2,7,0,5,0,0]), [3,5]),
             ndarray(new Int32Array([0,2,2,5,6,0]), [2,3]),
             ndarray(new Int32Array([1,0]), [1,2]),
             ndarray(new Int32Array([1]), [1,1])]

  var p = pyramid.detail(data, pyramid.adjunction)
  
  t.equal(p.length, ref.length, "number of levels")
  for(var i=0; i<Math.min(p.length, ref.length); i++) {
    t.ok(ops.equals(p[i], ref[i]), "level " + i)
  }

  t.end()
})

test("adjunction detail pyramid reconstruction", function(t) {
  var data = ndarray(new Int32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5])
  var ref = [data, ndarray(new Int32Array([2,4,3,7,8,1]), [2,3]), ndarray(new Int32Array([2,1]), [1,2]), ndarray(new Int32Array([1]), [1,1])]

  var p = pyramid.detail(data, pyramid.adjunction)
  pyramid.reconstruct(p, pyramid.adjunction)
  
  t.equal(p.length, ref.length, "number of levels")
  for(var i=0; i<Math.min(p.length, ref.length); i++) {
    t.ok(ops.equals(p[i], ref[i]), "level " + i)
  }

  t.end()
})

test("SunMaragos pyramid", function(t) {
  var data = ndarray(new Int32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5])
  var ref = [data, ndarray(new Int32Array([11,6,3,2,7,1]), [2,3]), ndarray(new Int32Array([11,3]), [1,2]), ndarray(new Int32Array([11]), [1,1])]

  var p = pyramid(data, pyramid.SunMaragos)

  t.equal(p.length, ref.length, "number of levels")
  for(var i=0; i<Math.min(p.length, ref.length); i++) {
    t.ok(ops.equals(p[i], ref[i]), "level " + i)
  }

  t.end()
})

test("binomial pyramid", function(t) {
  // TODO: Perhaps test behaviour on integer data?
  var data = ndarray(new Float32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5])
  var ref = [data,
             ndarray(new Float32Array([4.37890625,5.97265625,2.74609375,4.6328125,6.265625,2.3359375]), [2,3]),
             ndarray(new Float32Array([2.102508544921875,1.731719970703125]), [1,2]),
             ndarray(new Float32Array([0.4580140113830566]), [1,1])]

  var p = pyramid(data, pyramid.binomial)

  t.equal(p.length, ref.length, "number of levels")
  for(var i=0; i<Math.min(p.length, ref.length); i++) {
    t.ok(almostEqual(p[i], ref[i], 1e-6, 1e-6), "level " + i)
  }

  t.end()
})

test("binomial pyramid reconstruction", function(t) {
  // TODO: Perhaps test behaviour on integer data?
  var p = [ndarray(new Float32Array([7.444702148437500,5.43701171875000,1.27612304687500,8.192871093750000,0.578491210937500,-2.144287109375000,9.68750000000000,3.52978515625000,-0.330078125000000,2.329345703125000,10.302795410156250,2.26611328125000,8.15100097656250,4.229492187500000,-1.252136230468750]), [3,5]),
           ndarray(new Float32Array([3.033896446228027,4.534820556640625,1.574891090393066,3.736139297485352,5.307067871093750,1.555135726928711]), [2,3]),
           ndarray(new Float32Array([1.8448756635189056,1.5599647164344788]), [1,2]),
           ndarray(new Float32Array([0.4580140113830566]), [1,1])]
  var ref = [ndarray(new Float32Array([11,10,6,12,3,2,15,9,4,5,14,7,13,8,1]), [3,5]),
             ndarray(new Float32Array([4.37890625,5.97265625,2.74609375,4.6328125,6.265625,2.3359375]), [2,3]),
             ndarray(new Float32Array([2.102508544921875,1.731719970703125]), [1,2]),
             ndarray(new Float32Array([0.4580140113830566]), [1,1])]

  pyramid.reconstruct(p, pyramid.binomial)

  t.equal(p.length, ref.length, "number of levels")
  for(var i=0; i<Math.min(p.length, ref.length); i++) {
    t.ok(almostEqual(p[i], ref[i], 1e-6, 1e-6), "level " + i)
  }

  t.end()
})
