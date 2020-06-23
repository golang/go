// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5172: spurious warn about type conversion on broken type inside go and defer

package main

type foo struct {
	x bar // ERROR "undefined"
}

type T struct{}

func (t T) Bar() {}

func main() {
	var f foo
	go f.bar()    // ERROR "undefined"
	defer f.bar() // ERROR "undefined"

	t := T{1} // ERROR "too many values"
	go t.Bar()
}
