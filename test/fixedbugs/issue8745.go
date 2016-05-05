// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that the error says s[2] is a byte, not a uint8.

package p

func f(s string) {
	var _ float64 = s[2] // ERROR "cannot use.*type byte.*as type float64"
}
