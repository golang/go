// $G $F.go && $L $F.$A && ./$A.out 2>&1 | cmp - $D/$F.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	print("hello, world\n")
}
