// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T int

func (T) m() {} // GCCGO_ERROR "previous"
func (T) m() {} // ERROR "T[.]m redeclared|redefinition"

func (*T) p() {} // GCCGO_ERROR "previous"
func (*T) p() {} // ERROR "[(][*]T[)][.]p redeclared|redefinition|redeclared"
