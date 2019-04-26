// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nilfunc

func F() {}

func Comparison() {
	if F == nil { // ERROR "comparison of function F == nil is always false"
		panic("can't happen")
	}
}
