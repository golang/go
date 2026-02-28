// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 8311.
// error for x++ should say x++ not x += 1

package p

func f() {
	var x []byte
	x++ // ERROR "invalid operation: x[+][+]|non-numeric type"

}
