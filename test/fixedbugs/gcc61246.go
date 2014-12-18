// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR61246: Switch conditions could be untyped, causing an ICE when the
// conditions were lowered into temporaries.
// This is a reduction of a program reported by GoSmith.

package main

func main() {
	switch 1 != 1 {
	default:
	}
}
