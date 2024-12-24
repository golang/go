// -gotypesalias=1

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[P any] struct{}

func (T[P]) m1()

type A1 = T // ERROR "cannot use generic type"

func (A1[P]) m2() {} // don't report a follow-on error on A1

type A2 = T[int]

func (A2 /* ERROR "cannot define new methods on instantiated type T[int]" */) m3()   {}
func (_ A2 /* ERROR "cannot define new methods on instantiated type T[int]" */) m4() {}

func (T[int]) m5()                                       {} // int is the type parameter name, not an instantiation
func (T[* /* ERROR "must be an identifier" */ int]) m6() {} // syntax error
