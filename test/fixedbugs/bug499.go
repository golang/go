// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo got confused when a type was used both for a map bucket type
// and for a map key type.

package main

func main() {
	_ = make(map[byte]byte)
	_ = make(map[[8]byte]chan struct{})
}
