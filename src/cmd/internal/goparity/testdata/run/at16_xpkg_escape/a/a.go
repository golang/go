// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "unsafe"

var Saved []byte

//go:noinline
func StoreParam(s []byte) {
	Saved = s
}

var SavedBranch []byte

//go:noinline
func LeakBranch(s []byte, keep bool) {
	if keep {
		SavedBranch = s
	}
}

//go:noinline
func RetParam(s []byte) []byte {
	return s
}

//go:noinline
func StoreInto(p *Pointee, v int) {
	p.V = v
}

type Pointee struct {
	V int
}

//go:noinline
func LeakClosure(s []byte) func() byte {
	return func() byte {
		return s[0]
	}
}

//go:noinline
func RoundTrip(s []byte) []byte {
	if len(s) == 0 {
		return nil
	}
	p := unsafe.Pointer(&s[0])
	return unsafe.Slice((*byte)(p), len(s))
}

//go:noinline
func SetV(p *int, v int) {
	*p = v
}

type I interface {
	M() int
}

type E interface {
	I
	N() int
}

type GI[T any] interface {
	Get() T
}

type U interface {
	M() int
	u() int
}

type Empty = any

type impl struct {
	v int
}

func (e impl) M() int {
	return e.v
}

func (e impl) N() int {
	return e.v + 1
}

func (e impl) u() int {
	return e.v + 2
}

func MakeE(v int) E {
	return impl{v: v}
}

func MakeEmpty(v int) Empty {
	return impl{v: v}
}

func MakeU(v int) U {
	return impl{v: v}
}

type gimpl[T any] struct {
	v T
}

func (g gimpl[T]) Get() T {
	return g.v
}

func MakeGI(v int) GI[int] {
	return gimpl[int]{v: v}
}
