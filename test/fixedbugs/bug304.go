// $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Caused a gccgo crash on compilation.
// bug304.go: In function ‘p.f’:
// bug304.go:15:2: internal compiler error: in copy_tree_r, at tree-inline.c:4114

package p
type S struct {
	v interface{}
}
func g(e interface{}) { }
func f(s S) {
	g(s.v.(*int))
}
