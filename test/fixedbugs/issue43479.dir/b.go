// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

var Here = a.New()
var Dir = Here.Dir

type T = struct {
	a.Here
	a.I
}

var X T

// Test exporting the type of method values for anonymous structs with
// promoted methods.
var A = a.A
var B = a.B
var C = a.C
var D = a.D
var E = a.E
var F = a.F
var G = (*a.T).Dir
var H = a.T.Dir
var I = a.X.Dir
var J = (*a.T).M
var K = a.T.M
var L = a.X.M
var M = (*T).Dir
var N = T.Dir
var O = X.Dir
var P = (*T).M
var Q = T.M
var R = X.M
