// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package p

func f(...int) {}

func g() {
	var x []int
	f(x, x...) // ERROR "have \(\[\]int, \.\.\.int\)|too many arguments"
}
