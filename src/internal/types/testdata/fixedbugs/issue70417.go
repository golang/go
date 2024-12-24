// -gotypesalias=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[P any] struct{}

// A0
type A0 = T[int]
type B0 = *T[int]

func (A0 /* ERROR "cannot define new methods on instantiated type T[int]" */) m()  {}
func (*A0 /* ERROR "cannot define new methods on instantiated type T[int]" */) m() {}
func (B0 /* ERROR "cannot define new methods on instantiated type T[int]" */) m()  {}

// A1
type A1[P any] = T[P]
type B1[P any] = *T[P]

func (A1 /* ERROR "cannot define new methods on generic alias type A1[P any]" */ [P]) m()  {}
func (*A1 /* ERROR "cannot define new methods on generic alias type A1[P any]" */ [P]) m() {}
func (B1 /* ERROR "cannot define new methods on generic alias type B1[P any]" */ [P]) m()  {}

// A2
type A2[P any] = T[int]
type B2[P any] = *T[int]

func (A2 /* ERROR "cannot define new methods on generic alias type A2[P any]" */ [P]) m()  {}
func (*A2 /* ERROR "cannot define new methods on generic alias type A2[P any]" */ [P]) m() {}
func (B2 /* ERROR "cannot define new methods on generic alias type B2[P any]" */ [P]) m()  {}

// A3
type A3 = T[int]
type B3 = *T[int]

func (A3 /* ERROR "cannot define new methods on instantiated type T[int]" */) m()  {}
func (*A3 /* ERROR "cannot define new methods on instantiated type T[int]" */) m() {}
func (B3 /* ERROR "cannot define new methods on instantiated type T[int]" */) m()  {}

// A4
type A4 = T  // ERROR "cannot use generic type T[P any] without instantiation"
type B4 = *T // ERROR "cannot use generic type T[P any] without instantiation"

func (A4[P]) m1()  {} // don't report a follow-on error on A4
func (*A4[P]) m2() {} // don't report a follow-on error on A4
func (B4[P]) m3()  {} // don't report a follow-on error on B4

// instantiation in the middle of an alias chain
type S struct{}
type C0 = S
type C1[P any] = C0
type C2 = *C1[int]

func (C2 /* ERROR "cannot define new methods on instantiated type C1[int]" */) m()  {}
func (*C2 /* ERROR "cannot define new methods on instantiated type C1[int]" */) m() {}
