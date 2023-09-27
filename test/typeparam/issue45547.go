// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[T any]() (f, g T) { return f, g }

// Tests for generic function instantiation on the right hande side of multi-value
// assignments.

func g() {
	// Multi-value assignment within a function
	var _, _ = f[int]()
}

// Multi-value assignment outside a function.
var _, _ = f[int]()
