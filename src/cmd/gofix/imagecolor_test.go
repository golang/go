// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(colorTests, imagecolor)
}

var colorTests = []testCase{
	{
		Name: "color.0",
		In: `package main

import (
	"image"
)

var (
	_ image.Image
	_ image.RGBA
	_ image.Black
	_ image.Color
	_ image.ColorModel
	_ image.ColorModelFunc
	_ image.PalettedColorModel
	_ image.RGBAColor
	_ image.RGBA64Color
	_ image.NRGBAColor
	_ image.NRGBA64Color
	_ image.AlphaColor
	_ image.Alpha16Color
	_ image.GrayColor
	_ image.Gray16Color
)

func f() {
	_ = image.RGBAColorModel
	_ = image.RGBA64ColorModel
	_ = image.NRGBAColorModel
	_ = image.NRGBA64ColorModel
	_ = image.AlphaColorModel
	_ = image.Alpha16ColorModel
	_ = image.GrayColorModel
	_ = image.Gray16ColorModel
}
`,
		Out: `package main

import (
	"image"
	"image/color"
)

var (
	_ image.Image
	_ image.RGBA
	_ image.Black
	_ color.Color
	_ color.Model
	_ color.ModelFunc
	_ color.Palette
	_ color.RGBA
	_ color.RGBA64
	_ color.NRGBA
	_ color.NRGBA64
	_ color.Alpha
	_ color.Alpha16
	_ color.Gray
	_ color.Gray16
)

func f() {
	_ = color.RGBAModel
	_ = color.RGBA64Model
	_ = color.NRGBAModel
	_ = color.NRGBA64Model
	_ = color.AlphaModel
	_ = color.Alpha16Model
	_ = color.GrayModel
	_ = color.Gray16Model
}
`,
	},
	{
		Name: "color.1",
		In: `package main

import (
	"fmt"
	"image"
)

func f() {
	fmt.Println(image.RGBAColor{1, 2, 3, 4}.RGBA())
}
`,
		Out: `package main

import (
	"fmt"
	"image/color"
)

func f() {
	fmt.Println(color.RGBA{1, 2, 3, 4}.RGBA())
}
`,
	},
	{
		Name: "color.2",
		In: `package main

import "image"

var c *image.ColorImage = image.NewColorImage(nil)
`,
		Out: `package main

import "image"

var c *image.Uniform = image.NewUniform(nil)
`,
	},
}
