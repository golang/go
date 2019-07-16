// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo crashed compiling this code due to failing to finalize
// interfaces in the right order.

package p

type s1 struct {
	f m
	I
}

type m interface {
	Mm(*s2)
}

type s2 struct {
	*s1
}

type I interface {
	MI()
}
