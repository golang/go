// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func out_escapes() (x int, p *int) {
	p = &x;	// ERROR "address of out parameter"
	return;
}

func out_escapes_2() (x int, p *int) {
	return 2, &x;	// ERROR "address of out parameter"
}

