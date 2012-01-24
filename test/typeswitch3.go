// errchk $G -e $D/$F.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io"
)

type I interface {
	M()
}

func main(){
	var x I
	switch x.(type) {
	case string:	// ERROR "impossible"
		println("FAIL")
	}
	
	// Issue 2700: if the case type is an interface, nothing is impossible
	
	var r io.Reader
	
	_, _ = r.(io.Writer)
	
	switch r.(type) {
	case io.Writer:
	}
}


