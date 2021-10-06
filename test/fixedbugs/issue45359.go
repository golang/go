// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() {
	var i, j int
	var b bool
	i = -(i &^ i)
	for 1>>uint(i) == 0 {
		_ = func() {
			i, b = 0, true
		}
		_ = b
		i %= j
	}
}
