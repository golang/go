// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that error message regarding := appears on
// correct line (and not on the line of the 2nd :=).

package p

func f() {
    select {
    case x, x := <-func() chan int { // ERROR "x repeated on left side of :=|redefinition|declared and not used"
            c := make(chan int)
            return c
    }():
    }
}
