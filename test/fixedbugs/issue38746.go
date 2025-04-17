// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var g *uint64

func main() {
	var v uint64
	g = &v
	v &^= (1 << 31)
	v |= 1 << 63
	v &^= (1 << 63)
}
