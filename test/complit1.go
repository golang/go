// errchk $G -e $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var m map[int][3]int
func f() [3]int

func fp() *[3]int
var mp map[int]*[3]int

var (
	_ = [3]int{1,2,3}[:]  // ERROR "slice of unaddressable value"
	_ = m[0][:]  // ERROR "slice of unaddressable value"
	_ = f()[:]  // ERROR "slice of unaddressable value"
	
	// these are okay because they are slicing a pointer to an array
	_ = (&[3]int{1,2,3})[:]
	_ = mp[0][:]
	_ = fp()[:]
)