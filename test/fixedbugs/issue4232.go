// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 4232
// issue 7200

package p

func f() {
	var a [10]int
	_ = a[-1]  // ERROR "invalid array index -1|index out of bounds"
	_ = a[-1:] // ERROR "invalid slice index -1|index out of bounds"
	_ = a[:-1] // ERROR "invalid slice index -1|index out of bounds"
	_ = a[10]  // ERROR "invalid array index 10|index out of bounds"
	_ = a[9:10]
	_ = a[10:10]
	_ = a[9:12]            // ERROR "invalid slice index 12|index out of bounds"
	_ = a[11:12]           // ERROR "invalid slice index 11|index out of bounds"
	_ = a[1<<100 : 1<<110] // ERROR "overflows int" "invalid slice index 1 << 100|index out of bounds"

	var s []int
	_ = s[-1]  // ERROR "invalid slice index -1|index out of bounds"
	_ = s[-1:] // ERROR "invalid slice index -1|index out of bounds"
	_ = s[:-1] // ERROR "invalid slice index -1|index out of bounds"
	_ = s[10]
	_ = s[9:10]
	_ = s[10:10]
	_ = s[9:12]
	_ = s[11:12]
	_ = s[1<<100 : 1<<110] // ERROR "overflows int" "invalid slice index 1 << 100|index out of bounds"

	const c = "foofoofoof"
	_ = c[-1]  // ERROR "invalid string index -1|index out of bounds"
	_ = c[-1:] // ERROR "invalid slice index -1|index out of bounds"
	_ = c[:-1] // ERROR "invalid slice index -1|index out of bounds"
	_ = c[10]  // ERROR "invalid string index 10|index out of bounds"
	_ = c[9:10]
	_ = c[10:10]
	_ = c[9:12]            // ERROR "invalid slice index 12|index out of bounds"
	_ = c[11:12]           // ERROR "invalid slice index 11|index out of bounds"
	_ = c[1<<100 : 1<<110] // ERROR "overflows int" "invalid slice index 1 << 100|index out of bounds"

	var t string
	_ = t[-1]  // ERROR "invalid string index -1|index out of bounds"
	_ = t[-1:] // ERROR "invalid slice index -1|index out of bounds"
	_ = t[:-1] // ERROR "invalid slice index -1|index out of bounds"
	_ = t[10]
	_ = t[9:10]
	_ = t[10:10]
	_ = t[9:12]
	_ = t[11:12]
	_ = t[1<<100 : 1<<110] // ERROR "overflows int" "invalid slice index 1 << 100|index out of bounds"
}
