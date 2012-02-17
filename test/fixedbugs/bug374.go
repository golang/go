// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1556
package foo

type I interface {
	m() int
}

type T int

var _ I = T(0)	// GCCGO_ERROR "incompatible"

func (T) m(buf []byte) (a int, b xxxx) {  // ERROR "xxxx"
	return 0, nil
}
