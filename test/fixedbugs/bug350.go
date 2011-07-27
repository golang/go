// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T int

func (T) m() {}
func (T) m() {} // ERROR "T[.]m redeclared"

func (*T) p() {}
func (*T) p() {} // ERROR "[(][*]T[)][.]p redeclared"
