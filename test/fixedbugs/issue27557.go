// errorcheck -0 -l -m

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var sink interface{}

func _() {
	var t T
	f := t.noescape // ERROR "t.noescape does not escape"
	f()
}

func _() {
	var t T       // ERROR "moved to heap"
	f := t.escape // ERROR "t.escape does not escape"
	f()
}

func _() {
	var t T        // ERROR "moved to heap"
	f := t.returns // ERROR "t.returns does not escape"
	sink = f()
}

type T struct{}

func (t *T) noescape()   {}           // ERROR "t does not escape"
func (t *T) escape()     { sink = t } // ERROR "leaking param: t$"
func (t *T) returns() *T { return t } // ERROR "leaking param: t to result ~r0 level=0"

func (t *T) recursive() { // ERROR "leaking param: t$"
	sink = t

	var t2 T          // ERROR "moved to heap"
	f := t2.recursive // ERROR "t2.recursive does not escape"
	f()
}
