// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	const a uint64 = 10
	var _ int64 = a // ERROR "convert|cannot|incompatible"
}
