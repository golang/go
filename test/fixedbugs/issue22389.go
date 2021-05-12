// errorcheck -d=panic

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Foo struct{}

func (f *Foo) Call(cb func(*Foo)) {
	cb(f)
}

func main() {
	f := &Foo{}
	f.Call(func(f) {}) // ERROR "f .*is not a type"
}
