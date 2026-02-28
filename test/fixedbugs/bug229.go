// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

func main() {
	var t testing.T

	// make sure error mentions that
	// name is unexported, not just "name not found".

	t.common.name = nil // ERROR "unexported|undefined"

	println(testing.anyLowercaseName("asdf")) // ERROR "unexported|undefined"
}
