// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T interface {
	F1(i int) (i int) // ERROR "duplicate argument i|redefinition|previous"
	F2(i, i int) // ERROR "duplicate argument i|redefinition|previous"
	F3() (i, i int) // ERROR "duplicate argument i|redefinition|previous"
}

type T1 func(i, i int) // ERROR "duplicate argument i|redefinition|previous"
type T2 func(i int) (i int) // ERROR "duplicate argument i|redefinition|previous"
type T3 func() (i, i int) // ERROR "duplicate argument i|redefinition|previous"

type R struct{}

func (i *R) F1(i int)         {} // ERROR "duplicate argument i|redefinition|previous"
func (i *R) F2() (i int)      {return 0} // ERROR "duplicate argument i|redefinition|previous"
func (i *R) F3(j int) (j int) {return 0} // ERROR "duplicate argument j|redefinition|previous"

func F1(i, i int)      {} // ERROR "duplicate argument i|redefinition|previous"
func F2(i int) (i int) {return 0} // ERROR "duplicate argument i|redefinition|previous"
func F3() (i, i int)   {return 0, 0} // ERROR "duplicate argument i|redefinition|previous"
