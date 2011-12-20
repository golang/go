// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(ycbcrTests, imageycbcr)
}

var ycbcrTests = []testCase{
	{
		Name: "ycbcr.0",
		In: `package main

import (
	"image/ycbcr"
)

func f() {
	_ = ycbcr.RGBToYCbCr
	_ = ycbcr.YCbCrToRGB
	_ = ycbcr.YCbCrColorModel
	var _ ycbcr.YCbCrColor
	var _ ycbcr.YCbCr
	var (
		_ ycbcr.SubsampleRatio = ycbcr.SubsampleRatio444
		_ ycbcr.SubsampleRatio = ycbcr.SubsampleRatio422
		_ ycbcr.SubsampleRatio = ycbcr.SubsampleRatio420
	)
}
`,
		Out: `package main

import (
	"image"
	"image/color"
)

func f() {
	_ = color.RGBToYCbCr
	_ = color.YCbCrToRGB
	_ = color.YCbCrModel
	var _ color.YCbCr
	var _ image.YCbCr
	var (
		_ image.YCbCrSubsampleRatio = image.YCbCrSubsampleRatio444
		_ image.YCbCrSubsampleRatio = image.YCbCrSubsampleRatio422
		_ image.YCbCrSubsampleRatio = image.YCbCrSubsampleRatio420
	)
}
`,
	},
}
