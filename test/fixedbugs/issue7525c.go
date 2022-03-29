// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7525: self-referential array types.

package main

var z struct { // GC_ERROR "initialization loop for z"
	e [cap(z.e)]int // GCCGO_ERROR "array bound|typechecking loop|invalid array"
}
