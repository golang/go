// echo bug395 is broken  # takes 90+ seconds to break
// # $G $D/$F.go || echo bug395

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
