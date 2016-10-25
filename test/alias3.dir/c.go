// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"./b"
	"bytes"
	"go/build"
	"math"
)

func f => b.F
func inlined => b.Inlined

var _ func(*context, a.Writer) = f

func Check() {
	if pi != math.Pi {
		panic(0)
	}

	var w writer
	b.F(new(context), w)
	f(new(build.Context), bytes.NewBuffer(nil))

	if !inlined() {
		panic(1)
	}

	if &default_ != &build.Default {
		panic(2)
	}

	if sin(1) != math.Sin(1) {
		panic(3)
	}

	var _ *limitedReader = new(limitedReader2)
}

// local aliases
const pi => b.Pi

type (
	context => b.Context // not an interface
	writer  => b.Writer  // interface
)

// different aliases may refer to the same original
type limitedReader => b.LimitedReader
type limitedReader2 => b.LimitedReader2

var default_ => b.Default
var default2 => b.Default2

func sin => b.Sin
func sin2 => b.Sin

func main() {
	a.Check()
	b.Check()
	Check()
}
