// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// size_t StrLen(_GoString_ s) {
// 	return _GoStringLen(s);
// }
import "C"

func main() {
	C.StrLen1() // ERROR HERE
}
