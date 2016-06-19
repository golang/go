// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	const fffd = "\uFFFD"

	// runtime.intstring used to convert int64 to rune without checking
	// for truncation.
	u := uint64(0x10001f4a9)
	big := string(u)
	if big != fffd {
		panic("big != bad")
	}

	// cmd/compile used to require integer constants to fit into an "int".
	const huge = string(1<<100)
	if huge != fffd {
		panic("huge != bad")
	}
}
