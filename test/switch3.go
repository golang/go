// errchk $G -e $D/$F.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main


type I interface {
       M()
}

func bad() {
	var i I
	var s string

	switch i {
	case s:  // ERROR "mismatched types string and I"
	}

	switch s {
	case i:  // ERROR "mismatched types I and string"
	}
}

func good() {
	var i interface{}
	var s string

	switch i {
	case s:
	}

	switch s {
	case i:
	}
}
