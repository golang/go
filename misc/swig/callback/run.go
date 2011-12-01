// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"swig/callback"
)

type GoCallback struct{}

func (p *GoCallback) Run() string {
	return "GoCallback.Run"
}

func main() {
	c := callback.NewCaller()
	cb := callback.NewCallback()

	c.SetCallback(cb)
	s := c.Call()
	fmt.Println(s)
	if s != "Callback::run" {
		panic(s)
	}
	c.DelCallback()

	cb = callback.NewDirectorCallback(&GoCallback{})
	c.SetCallback(cb)
	s = c.Call()
	fmt.Println(s)
	if s != "GoCallback.Run" {
		panic(s)
	}
	c.DelCallback()
	callback.DeleteDirectorCallback(cb)
}
