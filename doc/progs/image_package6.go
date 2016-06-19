// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
)

func main() {
	m0 := image.NewRGBA(image.Rect(0, 0, 8, 5))
	m1 := m0.SubImage(image.Rect(1, 2, 5, 5)).(*image.RGBA)
	fmt.Println(m0.Bounds().Dx(), m1.Bounds().Dx()) // prints 8, 4
	fmt.Println(m0.Stride == m1.Stride)             // prints true
}
