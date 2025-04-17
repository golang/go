// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case where a slice of a user-defined byte type (not uint8 or byte) is
// converted to a string.  Same for slice of runes.

package main

type MyByte byte

type MyRune rune

func main() {
	var y []MyByte
	_ = string(y)

	var z []MyRune
	_ = string(z)
}
