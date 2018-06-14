// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to call wrong methods; issue 1290.

package main

type S struct {
}
func (S) a() int{
	return 0
}
func (S) b() int{
	return 1
}

func main() {
	var i interface {
		b() int
		a() int
	} = S{}
	if i.a() != 0 {
		panic("wrong method called")
	}
	if i.b() != 1 {
		panic("wrong method called")
	}
}
