// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug372

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2355
package main

type T struct {}
func (T) m() string { return "T" }

type TT struct {
	T
	m func() string
}


func ff() string { return "ff" }

func main() {
	var tt TT
	tt.m = ff

	if tt.m() != "ff" {
		println(tt.m(), "!= \"ff\"")
	}
}
