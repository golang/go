// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"image/color"
)

func main() {
	m := image.NewRGBA(image.Rect(0, 0, 640, 480))
	m.Set(5, 5, color.RGBA{255, 0, 0, 255})
	fmt.Println(m.At(5, 5))
}
