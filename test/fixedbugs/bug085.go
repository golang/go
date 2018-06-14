// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package P

var x int

func foo() {
	print(P.x);  // ERROR "undefined"
}

/*
uetli:~/Source/go1/test/bugs gri$ 6g bug085.go
bug085.go:6: P: undefined
Bus error
*/

/* expected scope hierarchy (outermost to innermost)

universe scope (contains predeclared identifiers int, float32, int32, len, etc.)
"solar" scope (just holds the package name P so it can be found but doesn't conflict)
global scope (the package global scope)
local scopes (function scopes)
*/
