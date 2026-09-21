// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the bool checker.

package bool

func _() {
	var f, g func() int

	if v, w := f(), g(); v == w || v == w { // ERROR "redundant or: v == w || v == w"
	}
}
