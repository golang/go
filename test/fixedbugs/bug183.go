//errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T int

func f() {
	var x struct { T };
	var y struct { T T };
	x = y;	// ERROR "cannot|incompatible"
	_ = x;
}

type T1 struct { T }
type T2 struct { T T }

func g() {
	var x T1;
	var y T2;
	x = y;	// ERROR "cannot|incompatible"
	_ = x;
}

