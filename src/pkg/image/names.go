// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

// Colors from the HTML 4.01 specification: http://www.w3.org/TR/REC-html40/types.html#h-6.5
// These names do not necessarily match those from other lists, such as the X11 color names.
var (
	Aqua    = ColorImage{RGBAColor{0x00, 0xff, 0xff, 0xff}}
	Black   = ColorImage{RGBAColor{0x00, 0x00, 0x00, 0xff}}
	Blue    = ColorImage{RGBAColor{0x00, 0x00, 0xff, 0xff}}
	Fuchsia = ColorImage{RGBAColor{0xff, 0x00, 0xff, 0xff}}
	Gray    = ColorImage{RGBAColor{0x80, 0x80, 0x80, 0xff}}
	Green   = ColorImage{RGBAColor{0x00, 0x80, 0x00, 0xff}}
	Lime    = ColorImage{RGBAColor{0x00, 0xff, 0x00, 0xff}}
	Maroon  = ColorImage{RGBAColor{0x80, 0x00, 0x00, 0xff}}
	Navy    = ColorImage{RGBAColor{0x00, 0x00, 0x80, 0xff}}
	Olive   = ColorImage{RGBAColor{0x80, 0x80, 0x00, 0xff}}
	Red     = ColorImage{RGBAColor{0xff, 0x00, 0x00, 0xff}}
	Purple  = ColorImage{RGBAColor{0x80, 0x00, 0x80, 0xff}}
	Silver  = ColorImage{RGBAColor{0xc0, 0xc0, 0xc0, 0xff}}
	Teal    = ColorImage{RGBAColor{0x00, 0x80, 0x80, 0xff}}
	White   = ColorImage{RGBAColor{0xff, 0xff, 0xff, 0xff}}
	Yellow  = ColorImage{RGBAColor{0xff, 0xff, 0x00, 0xff}}

	// These synonyms are not in HTML 4.01.
	Cyan    = Aqua
	Magenta = Fuchsia
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
