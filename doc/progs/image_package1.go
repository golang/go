// cmpout

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
)

func main() {
	p := image.Point{2, 1}
	fmt.Println("X is", p.X, "Y is", p.Y)
}
