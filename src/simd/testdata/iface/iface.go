// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package iface

import (
	"simd"
)

// A SIMD-dependent type alias
type MyInt8s = simd.Int8s

func Generic[T haslen](x int) int {
	var v T
	return x + v.Len()
}

// VL = Generic[MyInt8s](1) doesn't currently work.
// TODO: automatically transform those initializers into what is done here instead.
var VL int

func init() {
	VL = Generic[MyInt8s](1)
}

// A struct dependent on SIMD
type VectorC struct {
	Field simd.Float32s
}

type Ftype func(x any) any

var Fvar Ftype

// A dependent function with a dependent signature
func (v *VectorC) MethodOfSimd() bool {
	return false
}

func (v VectorC) Data() simd.Float32s {
	return v.Field
}

func (v VectorC) Foo(x VectorC) VectorC {
	return VectorC{Field: v.Field.Add(x.Field)}
}

func (v VectorC) Bar(x VectorC) VectorC {
	return VectorC{Field: v.Field.Add(x.Field)}
}

type Vint interface {
	MethodOfSimd() bool
}

type haslen interface {
	Len() int
}

type HasFoo[T any] interface {
	Foo(x T) T
}

type HasBar interface {
	Bar(x VectorC) VectorC
}

//go:noinline
func MakeHasFoo[T HasFoo[T]](v T) HasFoo[T] {
	return v
}

func MakeHasBar(v VectorC) HasBar {
	return v
}

func VC(x simd.Float32s) VectorC {
	return VectorC{x}
}

type EmbedBar struct {
	HasBar
}

type EmbedFoo[T HasFoo[T]] struct {
	HasFoo[T]
}

func MakeHasEmbedFoo[T HasFoo[T]](v T) EmbedFoo[T] {
	return EmbedFoo[T]{MakeHasFoo[T](v)}
}

//go:noinline
func MakeHasEmbedBar(v VectorC) EmbedBar {
	return EmbedBar{MakeHasBar(v)}
}

type HasQux[T any] interface {
	Qux(x T) HasFoo[T]
}

type Q struct {
	q *Q
}

func (q *Q) Qux(v VectorC) HasFoo[VectorC] {
	return v
}
