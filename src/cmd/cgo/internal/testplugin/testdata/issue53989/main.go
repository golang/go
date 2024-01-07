// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 53989: the use of jump table caused a function
// from the plugin jumps in the middle of the function
// to the function with the same name in the main
// executable. As these two functions may be compiled
// differently as plugin needs to be PIC, this causes
// crash.

package main

import (
	"plugin"

	"testplugin/issue53989/p"
)

func main() {
	p.Square(7) // call the function in main executable

	p, err := plugin.Open("issue53989.so")
	if err != nil {
		panic(err)
	}
	f, err := p.Lookup("Square")
	if err != nil {
		panic(err)
	}
	f.(func(int))(7) // call the plugin one
}
