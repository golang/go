// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var x [3][3]int

func main() {
	for i := range 3 {
		x[i][i] = 0
	}
}
