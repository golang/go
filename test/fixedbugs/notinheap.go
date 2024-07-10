// errorcheck -+

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test type-checking errors for not-in-heap types.

//go:build cgo

package p

import "runtime/cgo"

type nih struct{ _ cgo.Incomplete }

type embed4 map[nih]int // ERROR "incomplete \(or unallocatable\) map key not allowed"

type embed5 map[int]nih // ERROR "incomplete \(or unallocatable\) map value not allowed"

type emebd6 chan nih // ERROR "chan of incomplete \(or unallocatable\) type not allowed"

type okay1 *nih

type okay2 []nih

type okay3 func(x nih) nih

type okay4 interface {
	f(x nih) nih
}
