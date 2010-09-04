// $G $D/$F.go && $L $F.$A && ! ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var a []int
	var ia interface{} = a
	var m = make(map[interface{}] int)
	m[ia] = 1
}
