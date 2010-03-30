// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG wrong result

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var x int = 1
	if x != 1 {
		print("found ", x, ", expected 1\n")
		panic("fail")
	}
	{
		var x int = x + 1
		if x != 2 {
			print("found ", x, ", expected 2\n")
			panic("fail")
		}
	}
	{
		x := x + 1
		if x != 2 {
			print("found ", x, ", expected 2\n")
			panic("fail")
		}
	}
}
