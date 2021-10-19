// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "a"

func main() {
	if a.F[int64]() != 8 {
		panic("bad")
	}
	if a.G[int8]() != 1 {
		panic("bad")
	}
	// TODO: enable once 47631 is fixed.
	//if a.H[int64]() != 8 {
	//	panic("bad")
	//}
}
