// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure error message for invalid conditions
// or tags are consistent with earlier Go versions.

package p

func _() {
	if a := 10 { // ERROR "a := 10 used as value"
	}

	for b := 10 { // ERROR "b := 10 used as value"
	}

	switch c := 10 { // ERROR "c := 10 used as value"
	}
}
