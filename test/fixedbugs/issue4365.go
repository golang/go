// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that fields hide promoted methods.
// https://golang.org/issue/4365

package main

type T interface {
        M()
}

type M struct{}

func (M) M() {}

type Foo struct {
        M
}

func main() {
        var v T = Foo{} // ERROR "has no methods|not a method|cannot use"
        _ = v
}
