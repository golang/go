// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	for i := 0; i < 1e5; i++ {
		x := new([100]byte)
		_ = x
	}
}
