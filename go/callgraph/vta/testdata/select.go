// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo() string
}

type J interface {
	I
}

type B struct {
	p string
}

func (b B) Foo() string { return b.p }

func Baz(b1, b2 B, c1 chan I, c2 chan J) {
	for {
		select {
		case c1 <- b1:
			print("b1")
		case c2 <- b2:
			print("b2")
		case <-c1:
			print("c1")
		case k := <-c2:
			print(k.Foo())
			return
		}
	}
}

// Relevant SSA:
// func Baz(b1 B, b2 B, c1 chan I, c2 chan J):
//   ...
//   t2 = *t0
//   t3 = make I <- B (t2)
//   t4 = *t1
//   t5 = make J <- B (t4)
//   t6 = select blocking [c1<-t3, c2<-t5, <-c1, <-c2] (index int, ok bool, I, J)
//   t7 = extract t6 #0
//   t8 = t7 == 0:int
//   if t8 goto 2 else 3
//         ...
//  8:
//   t15 = extract t6 #3
//   t16 = invoke t15.Foo()
//   t17 = print(t18)

// WANT:
// Local(t3) -> Channel(chan testdata.I)
// Local(t5) -> Channel(chan testdata.J)
// Channel(chan testdata.I) -> Local(t6[2])
// Channel(chan testdata.J) -> Local(t6[3])
