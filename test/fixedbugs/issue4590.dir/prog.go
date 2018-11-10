// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./pkg1"
	"./pkg2"
)

func main() {
	if pkg1.T != pkg2.T {
		panic("pkg1.T != pkg2.T")
	}
	if pkg1.U != pkg2.U {
		panic("pkg1.U != pkg2.U")
	}
	if pkg1.V != pkg2.V {
		panic("pkg1.V != pkg2.V")
	}
	if pkg1.W != pkg2.W {
		panic("pkg1.W != pkg2.W")
	}
}
