// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T func()

type I interface {
	f, g ();	// ERROR "syntax|signature"
}

type J interface {
	h T;  // ERROR "syntax|signature"
}

