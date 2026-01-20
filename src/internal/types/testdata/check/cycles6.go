// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

// Below are the pieces of syntax corresponding to functions which can produce a
// type T without first having a value of type T. Notice that each causes a
// value of type T to be passed to unsafe.Sizeof while T is incomplete.

// literal on type
type T0 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(T0{})]int
// literal on value                                                             (not applicable)
// literal on pointer                                                           (not applicable)

// call on type
type T1 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(T1(42))]int
// call on value
func f2() T2
type T2 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(f2())]int
// call on pointer                                                              (not applicable)

// assert on type
var i3 interface{}
type T3 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(i3.(T3))]int
// assert on value                                                              (not applicable)
// assert on pointer                                                            (not applicable)

// receive on type                                                              (not applicable)
// receive on value
func f4() <-chan T4
type T4 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(<-f4())]int
// receive on pointer                                                           (not applicable)

// star on type                                                                 (not applicable)
// star on value                                                                (not applicable)
// star on pointer
func f5() *T5
type T5 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(*f5())]int

// Below is additional syntax which interacts with incomplete types. Notice that
// each of the below falls into 1 of 3 cases:
//   1. It cannot produce a value of (incomplete) type T.
//   2. It can, but only because it already has a value of type T.
//   3. It can, but only because it performs an implicit dereference.

// select on type                                                               (case 1)
// select on value                                                              (case 2)
type T6 /* ERROR "invalid recursive type" */ struct {
	f T7
}
type T7 [unsafe.Sizeof(T6{}.f)]int
// select on pointer                                                            (case 3)
type T8 /* ERROR "invalid recursive type" */ struct {
	f T9
}
type T9 [unsafe.Sizeof(new(T8).f)]int

// slice on type                                                                (not applicable)
// slice on value                                                               (case 2)
type T10 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(T10{}[:])]int
// slice on pointer                                                             (case 3)
type T11 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(new(T11)[:])]int

// index on type                                                                (case 1)
// index on value                                                               (case 2)
type T12 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(T12{}[42])]int
// index on pointer                                                             (case 3)
type T13 /* ERROR "invalid recursive type" */ [unsafe.Sizeof(new(T13)[42])]int
// index on map                                                                            (case 1)
type T14 /* ERROR "invalid recursive type" */ [unsafe.Sizeof((*new(map[int]T14))[42])]int
