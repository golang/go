// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var B []bool
var N int

func f(p bool, m map[bool]bool) bool {
	var q bool
	_ = p || N&N < N || B[0] || B[0]
	return p && q && m[q]
}
