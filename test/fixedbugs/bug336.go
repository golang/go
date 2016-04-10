// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T1 struct {
	Next *T2
}

type T2 T1

type T3 struct {
	Next *T4
}

type T4 T5
type T5 T6
type T6 T7
type T7 T8
type T8 T9
type T9 T3

type T10 struct {
	x struct {
		y ***struct {
			z *struct {
				Next *T11
			}
		}
	}
}

type T11 T10

type T12 struct {
	F1 *T15
	F2 *T13
	F3 *T16
}

type T13 T14
type T14 T15
type T15 T16
type T16 T17
type T17 T12

// issue 1672
type T18 *[10]T19
type T19 T18

func main() {
	_ = &T1{&T2{}}
	_ = &T2{&T2{}}
	_ = &T3{&T4{}}
	_ = &T4{&T4{}}
	_ = &T5{&T4{}}
	_ = &T6{&T4{}}
	_ = &T7{&T4{}}
	_ = &T8{&T4{}}
	_ = &T9{&T4{}}
	_ = &T12{&T15{}, &T13{}, &T16{}}

	var (
		tn    struct{ Next *T11 }
		tz    struct{ z *struct{ Next *T11 } }
		tpz   *struct{ z *struct{ Next *T11 } }
		tppz  **struct{ z *struct{ Next *T11 } }
		tpppz ***struct{ z *struct{ Next *T11 } }
		ty    struct {
			y ***struct{ z *struct{ Next *T11 } }
		}
	)
	tn.Next = &T11{}
	tz.z = &tn
	tpz = &tz
	tppz = &tpz
	tpppz = &tppz
	ty.y = tpppz
	_ = &T10{ty}

	t19s := &[10]T19{}
	_ = T18(t19s)
}
