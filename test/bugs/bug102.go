// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: should not crash

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var b [0]byte;
	s := string(b);	// out of bounds trap
	if s != "" {
		panic("bad convert")
	}
	var b1 = [5]byte{'h', 'e', 'l', 'l', 'o'};
	if string(b1) != "hello" {
		panic("bad convert 1")
	}
	var b2 = new([]byte, 5)
	for i := 0; i < 5; i++ { b2[i] = b1[i] }
	if string(b2) != "hello" {
		panic("bad convert 2")
	}
}

