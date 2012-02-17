// run

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S string
type I int
type F float64

func (S) m() {}
func (I) m() {}
func (F) m() {}

func main() {
	c := make(chan interface {
		m()
	},
		10)
	c <- I(0)
	c <- F(1)
	c <- S("hi")
	<-c
	<-c
	<-c
}
