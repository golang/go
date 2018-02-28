// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct {
	f [1]int
}

func a() {
	_ = T // ERROR "type T is not an expression"
}

func b() {
	var v [len(T{}.f)]int // ok
	_ = v
}
