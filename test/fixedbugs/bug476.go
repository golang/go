// compile

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Logical operation on named boolean type returns the same type,
// supporting an implicit convertion to an interface type.  This used
// to crash gccgo.

package p

type B bool

func (b B) M() {}

type I interface {
	M()
}

func F(a, b B) I {
	return a && b
}
