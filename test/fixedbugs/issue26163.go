// compile -N -d=softfloat

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 26163: dead store generated in late opt messes
// up store chain calculation.

package p

var i int
var A = ([]*int{})[i]

var F func(float64, complex128) int
var C chan complex128
var B = F(1, 1+(<-C))
