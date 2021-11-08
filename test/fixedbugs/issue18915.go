// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure error message for invalid conditions
// or tags are consistent with earlier Go versions.

package p

func _() {
	if a := 10 { // ERROR "cannot use a := 10 as value|expected .*;|declared but not used"
	}

	for b := 10 { // ERROR "cannot use b := 10 as value|parse error|declared but not used"
	}

	switch c := 10 { // ERROR "cannot use c := 10 as value|expected .*;|declared but not used"
	}
}
