// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import (
	"./a"
	"bytes"
	"go/build"
	"io"
	"math"
)

func F => a.F
func Inlined => a.Inlined

var _ func(*Context, io.Writer) = a.F

// check aliases
func Check() {
	if Pi != math.Pi {
		panic(0)
	}

	var w Writer
	a.F(new(Context), w)
	F(new(build.Context), bytes.NewBuffer(nil))

	if !Inlined() {
		panic(1)
	}

	if &Default != &build.Default {
		panic(2)
	}

	if Sin(1) != math.Sin(1) {
		panic(3)
	}

	var _ *LimitedReader = new(LimitedReader2)
}

// re-export aliases
const Pi => a.Pi

type (
	Context => a.Context // not an interface
	Writer  => a.Writer  // interface
)

// different aliases may refer to the same original
type LimitedReader => a.LimitedReader
type LimitedReader2 => a.LimitedReader2

var Default => a.Default
var Default2 => a.Default2

func Sin => a.Sin
func Sin2 => a.Sin
