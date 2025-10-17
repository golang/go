// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gen

import "sync"

type Value interface {
	kind() *kind
	getOp() *op
}

type Word interface {
	Value
	isWord()
}

// wrap is an unfortunate necessity so that we can pass Value types around as
// values (not pointers), but still have generic functions that can construct a
// new Value. Ideally we would just have a method on Value to initialize its op,
// but that needs to have a non-pointer receiver to satisfy the interface and
// then it can't mutate the Value.
type wrap[T Value] interface {
	Value
	wrap(x *op) T
}

type kind struct {
	typ string
	reg regClass
}

type void struct {
	valAny
}

var voidKind = &kind{typ: "void", reg: regClassNone}

func (void) kind() *kind { return voidKind }

type Ptr[T Value] struct {
	valGP
}

// Ptr is a Word
var _ Word = Ptr[void]{}

var ptrKinds = sync.Map{} // *kind -> *kind

func (Ptr[T]) kind() *kind {
	var x T
	xk := x.kind()
	pk, ok := ptrKinds.Load(xk)
	if !ok {
		k := &kind{typ: "Ptr[" + x.kind().typ + "]", reg: regClassGP}
		pk, _ = ptrKinds.LoadOrStore(xk, k)
	}
	return pk.(*kind)
}

func (Ptr[T]) wrap(x *op) Ptr[T] {
	var y Ptr[T]
	y.initOp(x)
	return y
}

func (x Ptr[T]) AddConst(off int) (y Ptr[T]) {
	base := x.op
	for base.op == "addConst" {
		off += base.args[1].c.(int)
		base = base.args[0]
	}
	y.initOp(&op{op: "addConst", kind: y.kind(), args: []*op{base, imm(off)}})
	return y
}

func Deref[W wrap[T], T Value](ptr Ptr[W]) T {
	var off int
	base := ptr.op
	for base.op == "addConst" {
		off += base.args[1].c.(int)
		base = base.args[0]
	}

	var y W
	return y.wrap(&op{op: "deref", kind: y.kind(), args: []*op{base}, c: off})
}

type Array[T Value] struct {
	valAny
}

func ConstArray[T Value](vals []T, name string) (y Array[T]) {
	// TODO: This probably doesn't actually work because emitConst won't
	// understand vals.
	y.initOp(&op{op: "const", kind: y.kind(), c: vals, name: name})
	return y
}

func (Array[T]) kind() *kind {
	// TODO: Cache this like Ptr.kind.
	var x T
	return &kind{typ: "Array[" + x.kind().typ + "]", reg: regClassNone}
}

type valGP struct {
	valAny
}

func (valGP) isWord() {}

type valAny struct {
	*op
}

func (v *valAny) initOp(x *op) {
	if v.op != nil {
		panic("double init of val")
	}
	if x.kind == nil {
		panic("val missing kind")
	}
	v.op = x

	// Figure out this value's function.
	for _, arg := range x.args {
		if fn := arg.fn; fn != nil {
			fn.attach(x)
			break
		}
	}
}

func (v valAny) getOp() *op {
	return v.op
}
