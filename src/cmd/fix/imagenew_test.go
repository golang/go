// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(imagenewTests, imagenew)
}

var imagenewTests = []testCase{
	{
		Name: "imagenew.0",
		In: `package main

import (
	"image"
)

func f() {
	image.NewRGBA(1, 2)
	image.NewRGBA64(1, 2)
	image.NewNRGBA(1, 2)
	image.NewNRGBA64(1, 2)
	image.NewAlpha(1, 2)
	image.NewAlpha16(1, 2)
	image.NewGray(1, 2)
	image.NewGray16(1, 2)
	image.NewPaletted(1, 2, nil)
}
`,
		Out: `package main

import (
	"image"
)

func f() {
	image.NewRGBA(image.Rect(0, 0, 1, 2))
	image.NewRGBA64(image.Rect(0, 0, 1, 2))
	image.NewNRGBA(image.Rect(0, 0, 1, 2))
	image.NewNRGBA64(image.Rect(0, 0, 1, 2))
	image.NewAlpha(image.Rect(0, 0, 1, 2))
	image.NewAlpha16(image.Rect(0, 0, 1, 2))
	image.NewGray(image.Rect(0, 0, 1, 2))
	image.NewGray16(image.Rect(0, 0, 1, 2))
	image.NewPaletted(image.Rect(0, 0, 1, 2), nil)
}
`,
	},
}
