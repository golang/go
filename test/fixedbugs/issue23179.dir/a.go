// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Large struct {
	x [256]int
}

func F(x int, _ int, _ bool, _ Large) int {
	return x
}
