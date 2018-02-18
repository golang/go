// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A couple of aliases cases that gccgo incorrectly gave errors for.

package p

func F1() {
	type E = struct{}
	type X struct{}
	var x X
	var y E = x
	_ = y
}

func F2() {
	type E = struct{}
	type S []E
	type T []struct{}
	type X struct{}
	var x X
	s := S{E{}}
	t := T{struct{}{}}
	_ = append(s, x)
	_ = append(s, t[0])
	_ = append(s, t...)
}
