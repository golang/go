// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "plugin"

func F() int {
	p2, err := plugin.Open("issue22175_plugin2.so")
	if err != nil {
		panic(err)
	}
	g, err := p2.Lookup("G")
	if err != nil {
		panic(err)
	}
	return g.(func() int)()
}

func main() {}
