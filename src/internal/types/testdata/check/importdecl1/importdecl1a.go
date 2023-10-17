// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue 8969.

package importdecl1

import "go/ast"
import . "unsafe"

var _ Pointer // use dot-imported package unsafe

// Test cases for issue 23914.

type A interface {
	// Methods m1, m2 must be type-checked in this file scope
	// even when embedded in an interface in a different
	// file of the same package.
	m1() ast.Node
	m2() Pointer
}
