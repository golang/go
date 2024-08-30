// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4452. Used to print many errors, now just one.

package main

func main() {
	_ = [...]int(4) // ERROR "\[\.\.\.\].*outside of array literal|invalid use of \[\.\.\.\] array"
}
