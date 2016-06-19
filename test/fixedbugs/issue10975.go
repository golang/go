// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 10975: Returning an invalid interface would cause
// `internal compiler error: getinarg: not a func`.

package main

type I interface {
	int // ERROR "interface contains embedded non-interface int"
}

func New() I {
	return struct{}{}
}
