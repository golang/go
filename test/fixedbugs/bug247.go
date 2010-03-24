// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug247

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	const (
		Delta = 100 * 1e6
		Count = 10
	)
	_ = int64(Delta * Count)
	var i interface{} = Count
	j := i.(int)
	if j != Count {
		println("j=", j)
		panic("fail")
	}
}
