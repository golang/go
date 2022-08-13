// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _(x int) {
	_ = ~x    // unary ~ permitted but the type-checker will complain
}

func _(x int) {
	_ = x ~ x // ERROR "unexpected ~ at end of statement"
}
