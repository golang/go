// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19515: compiler panics on spilling int128 constant.

package x

type VScrollPanel struct {
	x, y int
}

type Color struct {
	R, G, B, A float32
}

func maxF(a, b float32) float32 {
	if a > b {
		return 0
	}
	return 1
}

type TransformMatrix [6]float32

type Paint struct {
	xform      TransformMatrix
	feather    float32
	innerColor Color
	outerColor Color
}

func BoxGradient(x, y, w, h, f float32, iColor, oColor Color) Paint {
	return Paint{
		xform:      TransformMatrix{9, 0, 0, 0, x, y},
		feather:    maxF(1.0, f),
		innerColor: iColor,
		outerColor: oColor,
	}
}

func (v *VScrollPanel) Draw() {
	x := float32(v.x)
	y := float32(v.y)

	BoxGradient(x+x-2, y-1, 0, 0, 0, Color{}, Color{})
	BoxGradient(x+y-2, y-1, 0, 0, 0, Color{}, Color{})
}

