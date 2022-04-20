// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

// Check that all methods of T are collected before
// determining the result type of m (which embeds
// all methods of T).

type T interface {
	m() interface {T}
	E
}

var _ int = T.m(nil).m().e()

type E interface {
	e() int
}

// Check that unresolved forward chains are followed
// (see also comment in resolver.go, checker.typeDecl).

var _ int = C.m(nil).m().e()

type A B

type B interface {
	m() interface{C}
	E
}

type C A

// Check that interface type comparison for identity
// does not recur endlessly.

type T1 interface {
	m() interface{T1}
}

type T2 interface {
	m() interface{T2}
}

func _(x T1, y T2) {
	// Checking for assignability of interfaces must check
	// if all methods of x are present in y, and that they
	// have identical signatures. The signatures recur via
	// the result type, which is an interface that embeds
	// a single method m that refers to the very interface
	// that contains it. This requires cycle detection in
	// identity checks for interface types.
	x = y
}

type T3 interface {
	m() interface{T4}
}

type T4 interface {
	m() interface{T3}
}

func _(x T1, y T3) {
	x = y
}

// Check that interfaces are type-checked in order of
// (embedded interface) dependencies (was issue 7158).

var x1 T5 = T7(nil)

type T5 interface {
	T6
}

type T6 interface {
	m() T7
}
type T7 interface {
	T5
}

// Actual test case from issue 7158.

func wrapNode() Node {
	return wrapElement()
}

func wrapElement() Element {
	return nil
}

type EventTarget interface {
	AddEventListener(Event)
}

type Node interface {
	EventTarget
}

type Element interface {
	Node
}

type Event interface {
	Target() Element
}

// Check that accessing an interface method too early doesn't lead
// to follow-on errors due to an incorrectly computed type set.

type T8 interface {
	m() [unsafe.Sizeof(T8.m /* ERROR undefined */ )]int
}

var _ = T8.m // no error expected here
