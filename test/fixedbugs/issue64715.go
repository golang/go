// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func boolInt32(b bool) int32 {
	if b {
		return 1
	}

	return 0
}

func f(left uint16, right int32) (r uint16) {
	return left >> right
}

var n = uint16(65535)

func main() {
	println(f(n, boolInt32(int64(n^n) > 1)))
}
