// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Foo interface { }

type T struct {}
func (t *T) foo() {}

func main() {
	t := new(T);
	var i interface {};
	f, ok := i.(Foo);
	_, _, _ = t, f, ok;
}
