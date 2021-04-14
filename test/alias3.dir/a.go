// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "go/build"

type (
	Float64 = float64
	Rune    = rune
)

type (
	Int       int
	IntAlias  = Int
	IntAlias2 = IntAlias
	S         struct {
		Int
		IntAlias
		IntAlias2
	}
)

type (
	Context = build.Context
)

type (
	I1 interface {
		M1(IntAlias2) Float64
		M2() Context
	}

	I2 = interface {
		M1(Int) float64
		M2() build.Context
	}
)

var i1 I1
var i2 I2 = i1
