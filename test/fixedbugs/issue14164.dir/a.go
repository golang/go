// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// F is an exported function, small enough to be inlined.
// It defines a local interface with an unexported method
// f, which will appear with a package-qualified method
// name in the export data.
func F(x interface{}) bool {
	_, ok := x.(interface {
		f()
	})
	return ok
}

// Like F but with the unexported interface method f
// defined via an embedded interface t. The compiler
// always flattens embedded interfaces so there should
// be no difference between F and G. Alas, currently
// G is not inlineable (at least via export data), so
// the issue is moot, here.
func G(x interface{}) bool {
	type t0 interface {
		f()
	}
	_, ok := x.(interface {
		t0
	})
	return ok
}

// Like G but now the embedded interface is declared
// at package level. This function is inlineable via
// export data. The export data representation is like
// for F.
func H(x interface{}) bool {
	_, ok := x.(interface {
		t1
	})
	return ok
}

type t1 interface {
	f()
}
