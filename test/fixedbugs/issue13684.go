// run

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that a label name matching a constant name
// is permitted.

package main

const labelname = 1

func main() {
	goto labelname
labelname:
}
