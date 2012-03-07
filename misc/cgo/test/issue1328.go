// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "testing"

// extern void BackIntoGo(void);
// void IntoC(void);
import "C"

//export BackIntoGo
func BackIntoGo() {
	x := 1

	for i := 0; i < 10000; i++ {
		xvariadic(x)
		if x != 1 {
			panic("x is not 1?")
		}
	}
}

func xvariadic(x ...interface{}) {
}

func test1328(t *testing.T) {
	C.IntoC()
}
