// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2276.

// Check that the error messages says 
//	bug381.go:29: unsafe.Alignof(0) not used
// and not
//	bug381.go:29: 4 not used

// Issue 2768: previously got
//    bug381.go:30: cannot use 3 (type time.Weekday) as type int in function argument
// want
//    bug381.go:30: cannot use time.Wednesday (type time.Weekday) as type int in function argument

package main

import (
	"time"
	"unsafe"
)

func f(int)

func main() {
	unsafe.Alignof(0) // ERROR "unsafe\.Alignof|value computed is not used"
	f(time.Wednesday) // ERROR "time.Wednesday|incompatible type"
}
