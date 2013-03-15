// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var T interface {
	F1(i int) (i int) // ERROR "duplicate argument i"
	F2(i, i int) // ERROR "duplicate argument i"
	F3() (i, i int) // ERROR "duplicate argument i"
}

var T1 func(i, i int) // ERROR "duplicate argument i"
var T2 func(i int) (i int) // ERROR "duplicate argument i"
var T3 func() (i, i int) // ERROR "duplicate argument i"
