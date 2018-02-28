// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR61204: Making temporaries for zero-sized types caused an ICE in gccgo.
// This is a reduction of a program reported by GoSmith.

package main

func main() {
	type t [0]int
	var v t
	v, _ = [0]int{}, 0
	_ = v
}
