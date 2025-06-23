// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func atom(s string) {
	if s == nil {	// ERROR "nil|incompatible"
		return;
	}
}

func main() {}

/*
bug047.go:4: fatal error: stringpool: not string
*/
