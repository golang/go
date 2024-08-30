// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	if a.InlinedFakeTrue() {
		panic("returned true was the real one")
	}
	if !a.InlinedFakeFalse() {
		panic("returned false was the real one")
	}
	if a.InlinedFakeNil() == nil {
		panic("returned nil was the real one")
	}
	a.Test()
}
