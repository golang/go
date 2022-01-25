// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo() string
}

type B struct {
	p string
}

func (b B) Foo() string { return b.p }

func Baz(m map[I]*I) {
	for i, v := range m {
		*v = B{p: i.Foo()}
	}
}

// Relevant SSA:
//  func Baz(m map[I]*I):
//   0:
//    t0 = range m
//         jump 1
//   1:
//    t1 = next t0
//    t2 = extract t1 #0
//    if t2 goto 2 else 3
//   2:
//    t3 = extract t1 #1
//    t4 = extract t1 #2
//    t5 = local B (complit)
//    t6 = &t5.p [#0]
//    t7 = invoke t3.Foo()
//    *t6 = t7
//    t8 = *t5
//    t9 = make I <- B (t8)
//    *t4 = t9
//    jump 1
//   3:
//    return

// WANT:
// MapKey(testdata.I) -> Local(t1[1])
// Local(t1[1]) -> Local(t3)
// MapValue(*testdata.I) -> Local(t1[2])
// Local(t1[2]) -> Local(t4), MapValue(*testdata.I)
// Local(t8) -> Local(t9)
// Local(t9) -> Local(t4)
// Local(t4) -> Local(t1[2])
