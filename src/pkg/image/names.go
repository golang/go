// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

var (
	// Black is an opaque black ColorImage.
	Black = ColorImage{RGBAColor{0x00, 0x00, 0x00, 0xff}}
	// White is an opaque white ColorImage.
	White = ColorImage{RGBAColor{0xff, 0xff, 0xff, 0xff}}
)

// A ColorImage is a practically infinite-sized Image of uniform Color.
// It implements both the Color and Image interfaces.
type ColorImage struct {
	C Color
}

func (c ColorImage) RGBA() (r, g, b, a uint32) {
	return c.C.RGBA()
}

func (c ColorImage) ColorModel() ColorModel {
	return ColorModelFunc(func(Color) Color { return c.C })
}

func (c ColorImage) Width() int { return 1e9 }

func (c ColorImage) Height() int { return 1e9 }

func (c ColorImage) At(x, y int) Color { return c.C }

// Opaque scans the entire image and returns whether or not it is fully opaque.
func (c ColorImage) Opaque() bool {
	_, _, _, a := c.C.RGBA()
	return a == 0xffff
}
