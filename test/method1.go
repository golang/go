// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct { }
func (t *T) M(int, string);
func (t *T) M(int, float) { }   // ERROR "redeclared"

func f(int, string);
func f(int, float) { }  // ERROR "redeclared"

func g(a int, b string);
func g(a int, c string);  // ERROR "names changed"

