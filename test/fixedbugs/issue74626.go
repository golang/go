// errorcheck -goexperiment fieldtrack

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Fooer interface {
	Foo() string
}

type FooImpl struct{}

//go:nointerface
func (FooImpl) Foo() string { return "foo" }

func toInterface[T Fooer](fooer T) Fooer {
	return fooer
}

func main() {
	var iface Fooer = toInterface(FooImpl{}) // ERROR "does not satisfy Fooer"
	iface.Foo()
}
