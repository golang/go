// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for http://code.google.com/p/go/issues/detail?id=700

package main

import "os"

func f() (e int) {
	_ = &e
	return 999
}

func main() {
	if f() != 999 {
		os.Exit(1)
	}
}
