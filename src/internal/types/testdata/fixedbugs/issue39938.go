// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// All but E2 and E5 provide an "indirection" and break infinite expansion of a type.
type E0[P any] []P
type E1[P any] *P
type E2[P any] struct{ _ P }
type E3[P any] struct{ _ *P }
type E5[P any] struct{ _ [10]P }

type T0 struct {
        _ E0[T0]
}

type T0_ struct {
        E0[T0_]
}

type T1 struct {
        _ E1[T1]
}

type T2 /* ERROR "invalid recursive type" */ struct {
        _ E2[T2]
}

type T3 struct {
        _ E3[T3]
}

type T4 /* ERROR "invalid recursive type" */ [10]E5[T4]

type T5 struct {
	_ E0[E2[T5]]
}

type T6 struct {
	_ E0[E2[E0[E1[E2[[10]T6]]]]]
}

type T7 struct {
	_ E0[[10]E2[E0[E2[E2[T7]]]]]
}

type T8 struct {
	_ E0[[]E2[E0[E2[E2[T8]]]]]
}

type T9 /* ERROR "invalid recursive type" */ [10]E2[E5[E2[T9]]]

type T10 [10]E2[E5[E2[func(T10)]]]
