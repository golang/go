// skip

// When issue 1909 is fixed, change from skip to compile.

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1909
// Would OOM due to exponential recursion on Foo's expanded methodset in nodefmt

package test

type Foo interface {
	Bar() interface {
		Foo
	}
	Baz() interface {
		Foo
	}
	Bug() interface {
		Foo
	}
}
