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
//   t0 = make I <- B (b1)
//   t1 = make J <- B (b2)
//   t2 = select blocking [c1<-t0, c2<-t1, <-c1, <-c2] (index int, ok bool, I, J)
//   t3 = extract t2 #0
//   t4 = t73== 0:int
//   if t4 goto 2 else 3
//         ...
//  8:
//   t12 = extract t2 #3
//   t13 = invoke t12.Foo()
//   t14 = print(t15)

// WANT:
// Local(t0) -> Channel(chan testdata.I)
// Local(t1) -> Channel(chan testdata.J)
// Channel(chan testdata.I) -> Local(t2[2])
// Channel(chan testdata.J) -> Local(t2[3])
