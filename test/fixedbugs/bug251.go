// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I1 interface { // GC_ERROR "invalid recursive type"
	m() I2
	I2
}

type I2 interface {
	I1 // GCCGO_ERROR "loop|interface"
}


var i1 I1 = i2
var i2 I2
var i2a I2 = i1
