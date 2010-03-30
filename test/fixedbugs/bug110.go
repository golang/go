// $G $D/$F.go && $L $F.$A || echo BUG: const bug

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const a = 0

func f() {
	const a = 5
}

func main() {
	if a != 0 {
		println("a=", a)
		panic("fail")
	}
}
