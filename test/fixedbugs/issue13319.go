// errorcheck

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(int, int) {
    switch x {
    case 1:
        f(1, g()   // ERROR "expecting \)|expecting comma or \)"
    case 2:
        f()
    case 3:
        f(1, g()   // ERROR "expecting \)|expecting comma or \)"
    }
}
