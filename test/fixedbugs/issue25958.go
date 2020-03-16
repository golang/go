// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Verify that the "must be receive" error for "case done:" appears
// on the line of the case clause, not the line of the done declaration.

func f(done chan struct{}) {
	select {
	case done: // ERROR "must be receive", "not used"
	case (chan struct{})(done): // ERROR "must be receive"
	}
}
