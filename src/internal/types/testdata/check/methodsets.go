// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package methodsets

type T0 struct {}

func (T0) v0() {}
func (*T0) p0() {}

type T1 struct {} // like T0 with different method names

func (T1) v1() {}
func (*T1) p1() {}

type T2 interface {
	v2()
	p2()
}

type T3 struct {
	T0
	*T1
	T2
}

// Method expressions
func _() {
	var (
		_ func(T0) = T0.v0
		_ = T0.p0 /* ERROR "invalid method expression T0.p0 (needs pointer receiver (*T0).p0)" */

		_ func (*T0) = (*T0).v0
		_ func (*T0) = (*T0).p0

		// T1 is like T0

		_ func(T2) = T2.v2
		_ func(T2) = T2.p2

		_ func(T3) = T3.v0
		_ func(T3) = T3.p0 /* ERROR "invalid method expression T3.p0 (needs pointer receiver (*T3).p0)" */
		_ func(T3) = T3.v1
		_ func(T3) = T3.p1
		_ func(T3) = T3.v2
		_ func(T3) = T3.p2

		_ func(*T3) = (*T3).v0
		_ func(*T3) = (*T3).p0
		_ func(*T3) = (*T3).v1
		_ func(*T3) = (*T3).p1
		_ func(*T3) = (*T3).v2
		_ func(*T3) = (*T3).p2
	)
}

// Method values with addressable receivers
func _() {
	var (
		v0 T0
		_ func() = v0.v0
		_ func() = v0.p0
	)

	var (
		p0 *T0
		_ func() = p0.v0
		_ func() = p0.p0
	)

	// T1 is like T0

	var (
		v2 T2
		_ func() = v2.v2
		_ func() = v2.p2
	)

	var (
		v4 T3
		_ func() = v4.v0
		_ func() = v4.p0
		_ func() = v4.v1
		_ func() = v4.p1
		_ func() = v4.v2
		_ func() = v4.p2
	)

	var (
		p4 *T3
		_ func() = p4.v0
		_ func() = p4.p0
		_ func() = p4.v1
		_ func() = p4.p1
		_ func() = p4.v2
		_ func() = p4.p2
	)
}

// Method calls with addressable receivers
func _() {
	var v0 T0
	v0.v0()
	v0.p0()

	var p0 *T0
	p0.v0()
	p0.p0()

	// T1 is like T0

	var v2 T2
	v2.v2()
	v2.p2()

	var v4 T3
	v4.v0()
	v4.p0()
	v4.v1()
	v4.p1()
	v4.v2()
	v4.p2()

	var p4 *T3
	p4.v0()
	p4.p0()
	p4.v1()
	p4.p1()
	p4.v2()
	p4.p2()
}

// Method values with value receivers
func _() {
	var (
		_ func() = T0{}.v0
		_ func() = T0{}.p0 /* ERROR "cannot call pointer method p0 on T0" */

		_ func() = (&T0{}).v0
		_ func() = (&T0{}).p0

		// T1 is like T0

		// no values for T2

		_ func() = T3{}.v0
		_ func() = T3{}.p0 /* ERROR "cannot call pointer method p0 on T3" */
		_ func() = T3{}.v1
		_ func() = T3{}.p1
		_ func() = T3{}.v2
		_ func() = T3{}.p2

		_ func() = (&T3{}).v0
		_ func() = (&T3{}).p0
		_ func() = (&T3{}).v1
		_ func() = (&T3{}).p1
		_ func() = (&T3{}).v2
		_ func() = (&T3{}).p2
	)
}

// Method calls with value receivers
func _() {
	T0{}.v0()
	T0{}.p0 /* ERROR "cannot call pointer method p0 on T0" */ ()

	(&T0{}).v0()
	(&T0{}).p0()

	// T1 is like T0

	// no values for T2

	T3{}.v0()
	T3{}.p0 /* ERROR "cannot call pointer method p0 on T3" */ ()
	T3{}.v1()
	T3{}.p1()
	T3{}.v2()
	T3{}.p2()

	(&T3{}).v0()
	(&T3{}).p0()
	(&T3{}).v1()
	(&T3{}).p1()
	(&T3{}).v2()
	(&T3{}).p2()
}

// *T has no methods if T is an interface type
func issue5918() {
	var (
		err error
		_ = err.Error()
		_ func() string = err.Error
		_ func(error) string = error.Error

		perr = &err
		_ = perr.Error /* ERROR "type *error is pointer to interface, not interface" */ ()
		_ func() string = perr.Error /* ERROR "type *error is pointer to interface, not interface" */
		_ func(*error) string = (*error).Error /* ERROR "type *error is pointer to interface, not interface" */
	)

	type T *interface{ m() int }
	var (
		x T
		_ = (*x).m()
		_ = (*x).m

		_ = x.m /* ERROR "type T is pointer to interface, not interface" */ ()
		_ = x.m /* ERROR "type T is pointer to interface, not interface" */
		_ = T.m /* ERROR "type T is pointer to interface, not interface" */
	)
}
