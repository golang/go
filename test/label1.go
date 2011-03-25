// errchk $G -e $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Pass 2 label errors.

package main

var x int

func f() {
L1:
	for {
		if x == 0 {
			break L1
		}
		if x == 1 {
			continue L1
		}
		goto L1
	}

L2:
	select {
	default:
		if x == 0 {
			break L2
		}
		if x == 1 {
			continue L2 // ERROR "invalid continue label .*L2"
		}
		goto L2
	}

L3:
	switch {
	case x > 10:
		if x == 11 {
			break L3
		}
		if x == 12 {
			continue L3 // ERROR "invalid continue label .*L3"
		}
		goto L3
	}

L4:
	if true {
		if x == 13 {
			break L4 // ERROR "invalid break label .*L4"
		}
		if x == 14 {
			continue L4 // ERROR "invalid continue label .*L4"
		}
		if x == 15 {
			goto L4
		}
	}

L5:
	f()
	if x == 16 {
		break L5 // ERROR "invalid break label .*L5"
	}
	if x == 17 {
		continue L5 // ERROR "invalid continue label .*L5"
	}
	if x == 18 {
		goto L5
	}

	for {
		if x == 19 {
			break L1 // ERROR "invalid break label .*L1"
		}
		if x == 20 {
			continue L1 // ERROR "invalid continue label .*L1"
		}
		if x == 21 {
			goto L1
		}
	}
}
