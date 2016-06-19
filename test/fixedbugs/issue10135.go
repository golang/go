// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 10135: append a slice with zero-sized element used
// to always return a slice with the same data pointer as the
// old slice, even if it's nil, so this program used to panic
// with nil pointer dereference because after append, s is a
// slice with nil data pointer but non-zero len and cap.

package main

type empty struct{}

func main() {
	var s []empty

	s = append(s, empty{})

	for _, v := range s {
		_ = v
	}
}
