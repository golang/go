// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

func Foo() {}

func Do(b bool) func() {
	if b {
		return Foo
	}
	return func() {}
}

func Finish(h func()) {
	h()
}

func Baz(b bool) {
	Finish(Do(b))
}

// Relevant SSA:
// func Baz(b bool):
//   t0 = Do(b)
//   t1 = Finish(t0)
//   return

// func Do(b bool) func():
//   if b goto 1 else 2
//  1:
//   return Foo
//  2:
//   return Do$1

// func Finish(h func()):
//   t0 = h()
//   return

// WANT:
// Baz: Do(b) -> Do; Finish(t0) -> Finish
// Finish: h() -> Do$1, Foo
