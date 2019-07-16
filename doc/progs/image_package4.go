// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
)

func main() {
	r := image.Rect(0, 0, 4, 3).Intersect(image.Rect(2, 2, 5, 5))
	// Size returns a rectangle's width and height, as a Point.
	fmt.Printf("%#v\n", r.Size()) // prints image.Point{X:2, Y:1}
}
