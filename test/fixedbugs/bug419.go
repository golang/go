// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1811.
// gccgo failed to compile this.

package p

type E interface{}

type I interface {
	E
	E
}
