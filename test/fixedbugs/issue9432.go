// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gc used to recurse infinitely when dowidth is applied
// to a broken recursive type again.
// See golang.org/issue/9432.
package p

type foo struct { // ERROR "invalid recursive type"
	bar  foo
	blah foo
}
