// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./bug"

type foo int

func (f *foo) Bar() {
}

func main() {
	bug.Foo(new(foo))
}
