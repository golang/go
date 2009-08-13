// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() int { }	// ERROR "return|control"
func g() (foo int) { }	// ERROR "return|control"
