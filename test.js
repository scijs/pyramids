"use strict"
var test = require('tape')
var pyramid = require("./index.js")
var ops = require('ndarray-ops')
var ndarray = require('ndarray')

// The references should not be considered as "perfect" (especially for SunMaragos), since fairly basic boundary conditions were assumed in these results.
// They are still useful for detecting regressions though.

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
