// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check various syntax errors with switches.

package main

func _() {
	switch {
	case 0; // ERROR "expecting := or = or : or comma"
	}

	switch {
	case 0; // ERROR "expecting := or = or : or comma"
	default:
	}

	switch {
	if x: // ERROR "expecting case or default or }"
	}
}
