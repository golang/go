// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 9634: Structs are incorrectly unpacked when passed as an argument
// to append.

package main

func main() {
	s := struct{
		t []int
		u int
	}{}
	_ = append(s, 0) // ERROR "must be a slice|must be slice"
}
