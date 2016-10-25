// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"bytes"
	"go/build"
	"io"
	"math"
)

func F(c *build.Context, w io.Writer) {}

func Inlined() bool { var w Writer; return w == nil }

func Check() {
	if Pi != math.Pi {
		panic(0)
	}

	var w Writer
	F(new(Context), w)
	F(new(build.Context), bytes.NewBuffer(nil))

	if &Default != &build.Default {
		panic(1)
	}

	if Sin(1) != math.Sin(1) {
		panic(2)
	}

	var _ *LimitedReader = new(LimitedReader2)
}

// export aliases
const Pi => math.Pi

type (
	Context => build.Context // not an interface
	Writer  => io.Writer     // interface
)

// different aliases may refer to the same original
type LimitedReader => io.LimitedReader
type LimitedReader2 => io.LimitedReader

var Default => build.Default
var Default2 => build.Default

func Sin => math.Sin
func Sin2 => math.Sin
