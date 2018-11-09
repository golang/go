// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "context"

// Return from main is handled specially.
// Since the program exits, there's no need to call cancel.
func main() {
	_, cancel := context.WithCancel(nil)
	if maybe {
		cancel()
	}
}

func notMain() {
	_, cancel := context.WithCancel(nil) // want "cancel function.*not used"

	if maybe {
		cancel()
	}
} // want "return statement.*reached without using the cancel"

var maybe bool
