// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// The purpose of this inherited test is unclear.

import "C"

const x = 1

var a, b = 1, 2

func F() {
}

func FAD(int, string) bool {
	C.malloc(3)
	return true
}
