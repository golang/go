// errorcheck -0 -d=tailcall=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Test that when generating wrappers for methods, we generate a tail call to the pointer version of
// the method, if that method is not inlineable. We use go:noinline here to force the non-inlineability
// condition.

//go:noinline
func (f *Foo) Get2Vals() [2]int { return [2]int{f.Val, f.Val + 1} }
func (f *Foo) Get3Vals() [3]int { return [3]int{f.Val, f.Val + 1, f.Val + 2} }

type Foo struct{ Val int }

type Bar struct { // ERROR "tail call emitted for the method \(\*Foo\).Get2Vals wrapper"
	int64
	*Foo // needs a method wrapper
	string
}

var i any

func init() {
	i = Bar{1, nil, "first"}
}
