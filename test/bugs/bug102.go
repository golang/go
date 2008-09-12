// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: should not crash

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var b [0]byte;
	s := string(b);	// out of bounds trap
}

