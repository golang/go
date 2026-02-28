// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "plugin"

func main() {
	p, err := plugin.Open("issue24351.so")
	if err != nil {
		panic(err)
	}
	f, err := p.Lookup("B")
	if err != nil {
		panic(err)
	}
	c := make(chan bool)
	f.(func(chan bool))(c)
	<-c
}
