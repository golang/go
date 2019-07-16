// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(x int) {
	_ = ~x    // ERROR "invalid character"
	_ = x ~ x // ERROR "invalid character" "unexpected x at end of statement"
}
