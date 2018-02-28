// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that method redeclarations are caught by the compiler.
// Does not compile.

package main

type T struct{}

func (t *T) M(int, string)  // GCCGO_ERROR "previous"
func (t *T) M(int, float64) {} // ERROR "redeclared|redefinition"

func (t T) H()  // GCCGO_ERROR "previous"
func (t *T) H() {} // ERROR "redeclared|redefinition"

func f(int, string)  // GCCGO_ERROR "previous"
func f(int, float64) {} // ERROR "redeclared|redefinition"

func g(a int, b string) // GCCGO_ERROR "previous"
func g(a int, c string) // ERROR "redeclared|redefinition"
