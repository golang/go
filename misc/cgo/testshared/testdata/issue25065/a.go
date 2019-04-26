// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package issue25065 has a type with a method that is
//  1) referenced in a method expression
//  2) not called
//  3) not converted to an interface
//  4) is a value method but the reference is to the pointer method
// These cases avoid the call to makefuncsym from typecheckfunc, but we
// still need to call makefuncsym somehow or the symbol will not be defined.
package issue25065

type T int

func (t T) M() {}

func F() func(*T) {
	return (*T).M
}
