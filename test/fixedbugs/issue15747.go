// errorcheck -0 -live -d=eagerwb

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 15747: liveness analysis was marking heap-escaped params live too much,
// and worse was using the wrong bitmap bits to do so.

// TODO(austin): This expects function calls to the write barrier, so
// we enable the legacy eager write barrier. Fix this once the
// buffered write barrier works on all arches.

package p

var global *[]byte

type Q struct{}

type T struct{ M string }

var b bool

func f1(q *Q, xx []byte) interface{} { // ERROR "live at call to newobject: xx$" "live at call to writebarrierptr: &xx$" "live at entry to f1: xx$"
	// xx was copied from the stack to the heap on the previous line:
	// xx was live for the first two prints but then it switched to &xx
	// being live. We should not see plain xx again.
	if b {
		global = &xx // ERROR "live at call to writebarrierptr: &xx$"
	}
	xx, _, err := f2(xx, 5) // ERROR "live at call to f2: &xx$" "live at call to writebarrierptr: err.data err.type$"
	if err != nil {
		return err
	}
	return nil
}

//go:noinline
func f2(d []byte, n int) (odata, res []byte, e interface{}) { // ERROR "live at entry to f2: d$"
	if n > len(d) {
		return d, nil, &T{M: "hello"} // ERROR "live at call to newobject: d" "live at call to writebarrierptr: d"
	}
	res = d[:n]
	odata = d[n:]
	return
}
