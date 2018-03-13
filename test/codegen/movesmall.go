// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func movesmall() {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7}
	copy(x[1:], x[:]) // arm64:-".*memmove"
}
