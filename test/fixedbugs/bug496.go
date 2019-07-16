// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo used to give an error:
// <built-in>: error: redefinition of ‘s$F$hash’
// <built-in>: note: previous definition of ‘s$F$hash’ was here
// <built-in>: error: redefinition of ‘s$F$equal’
// <built-in>: note: previous definition of ‘s$F$equal’ was here

package p

type T1 int

func (t T1) F() {
	type s struct {
		f string
	}
}

type T2 int

func (t T2) F() {
	type s struct {
		f string
	}
}
