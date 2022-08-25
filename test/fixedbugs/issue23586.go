// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we type-check deferred/go functions even
// if they are not called (a common error). Specifically,
// we don't want to see errors such as import or variable
// declared but not used.

package p

// TODO(gri) The "not used" errors should not be reported.

import (
	"fmt"  // ERROR "imported and not used"
	"math" // ERROR "imported and not used"
)

func f() {
	var i int // ERROR "i declared but not used"
	defer func() { fmt.Println() } // ERROR "must be function call"
	go func() { _ = math.Sin(0) }  // ERROR "must be function call"
	go func() { _ = i}             // ERROR "must be function call"
}
