// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
)

func main() {
	r := image.Rect(2, 1, 5, 5).Add(image.Pt(-4, -2))
	fmt.Println(r.Dx(), r.Dy(), image.Pt(0, 0).In(r)) // prints 3 4 true
}
