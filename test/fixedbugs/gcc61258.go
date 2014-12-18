// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR61258: gccgo crashed when deleting a zero-sized key from a map.

package main

func main() {
	delete(make(map[[0]bool]int), [0]bool{})
}
