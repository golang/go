// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"debug/proc"
	"exp/eval"
	"fmt"
)

// A RemoteMismatchError occurs when an operation that requires two
// identical remote processes is given different process.  For
// example, this occurs when trying to set a pointer in one process to
// point to something in another process.
type RemoteMismatchError string

func (e RemoteMismatchError) String() string { return string(e) }

// A ReadOnlyError occurs when attempting to set or assign to a
// read-only value.
type ReadOnlyError string

func (e ReadOnlyError) String() string { return string(e) }

// A maker is a function that converts a remote address into an
// interpreter Value.
type maker func(remote) eval.Value

type remoteValue interface {
	addr() remote
}

// remote represents an address in a remote process.
type remote struct {
	base proc.Word
	p    *Process
}

func (v remote) Get(a aborter, size int) uint64 {
	// TODO(austin) This variable might temporarily be in a
	// register.  We could trace the assembly back from the
	// current PC, looking for the beginning of the function or a
	// call (both of which guarantee that the variable is in
	// memory), or an instruction that loads the variable into a
	// register.
	//
	// TODO(austin) If this is a local variable, it might not be
	// live at this PC.  In fact, because the compiler reuses
	// slots, there might even be a different local variable at
	// this location right now.  A simple solution to both
	// problems is to include the range of PC's over which a local
	// variable is live in the symbol table.
	//
	// TODO(austin) We need to prevent the remote garbage
	// collector from collecting objects out from under us.
	var arr [8]byte
	buf := arr[0:size]
	_, err := v.p.Peek(v.base, buf)
	if err != nil {
		a.Abort(err)
	}
	return uint64(v.p.ToWord(buf))
}

func (v remote) Set(a aborter, size int, x uint64) {
	var arr [8]byte
	buf := arr[0:size]
	v.p.FromWord(proc.Word(x), buf)
	_, err := v.p.Poke(v.base, buf)
	if err != nil {
		a.Abort(err)
	}
}

func (v remote) plus(x proc.Word) remote { return remote{v.base + x, v.p} }

func tryRVString(f func(a aborter) string) string {
	var s string
	err := try(func(a aborter) { s = f(a) })
	if err != nil {
		return fmt.Sprintf("<error: %v>", err)
	}
	return s
}

/*
 * Bool
 */

type remoteBool struct {
	r remote
}

func (v remoteBool) String() string {
	return tryRVString(func(a aborter) string { return fmt.Sprintf("%v", v.aGet(a)) })
}

func (v remoteBool) Assign(t *eval.Thread, o eval.Value) {
	v.Set(t, o.(eval.BoolValue).Get(t))
}

func (v remoteBool) Get(t *eval.Thread) bool { return v.aGet(t) }

func (v remoteBool) aGet(a aborter) bool { return v.r.Get(a, 1) != 0 }

func (v remoteBool) Set(t *eval.Thread, x bool) {
	v.aSet(t, x)
}

func (v remoteBool) aSet(a aborter, x bool) {
	if x {
		v.r.Set(a, 1, 1)
	} else {
		v.r.Set(a, 1, 0)
	}
}

func (v remoteBool) addr() remote { return v.r }

func mkBool(r remote) eval.Value { return remoteBool{r} }

/*
 * Uint
 */

type remoteUint struct {
	r    remote
	size int
}

func (v remoteUint) String() string {
	return tryRVString(func(a aborter) string { return fmt.Sprintf("%v", v.aGet(a)) })
}

func (v remoteUint) Assign(t *eval.Thread, o eval.Value) {
	v.Set(t, o.(eval.UintValue).Get(t))
}

func (v remoteUint) Get(t *eval.Thread) uint64 {
	return v.aGet(t)
}

func (v remoteUint) aGet(a aborter) uint64 { return v.r.Get(a, v.size) }

func (v remoteUint) Set(t *eval.Thread, x uint64) {
	v.aSet(t, x)
}

func (v remoteUint) aSet(a aborter, x uint64) { v.r.Set(a, v.size, x) }

func (v remoteUint) addr() remote { return v.r }

func mkUint8(r remote) eval.Value { return remoteUint{r, 1} }

func mkUint16(r remote) eval.Value { return remoteUint{r, 2} }

func mkUint32(r remote) eval.Value { return remoteUint{r, 4} }

func mkUint64(r remote) eval.Value { return remoteUint{r, 8} }

func mkUint(r remote) eval.Value { return remoteUint{r, r.p.IntSize()} }

func mkUintptr(r remote) eval.Value { return remoteUint{r, r.p.PtrSize()} }

/*
 * Int
 */

type remoteInt struct {
	r    remote
	size int
}

func (v remoteInt) String() string {
	return tryRVString(func(a aborter) string { return fmt.Sprintf("%v", v.aGet(a)) })
}

func (v remoteInt) Assign(t *eval.Thread, o eval.Value) {
	v.Set(t, o.(eval.IntValue).Get(t))
}

func (v remoteInt) Get(t *eval.Thread) int64 { return v.aGet(t) }

func (v remoteInt) aGet(a aborter) int64 { return int64(v.r.Get(a, v.size)) }

func (v remoteInt) Set(t *eval.Thread, x int64) {
	v.aSet(t, x)
}

func (v remoteInt) aSet(a aborter, x int64) { v.r.Set(a, v.size, uint64(x)) }

func (v remoteInt) addr() remote { return v.r }

func mkInt8(r remote) eval.Value { return remoteInt{r, 1} }

func mkInt16(r remote) eval.Value { return remoteInt{r, 2} }

func mkInt32(r remote) eval.Value { return remoteInt{r, 4} }

func mkInt64(r remote) eval.Value { return remoteInt{r, 8} }

func mkInt(r remote) eval.Value { return remoteInt{r, r.p.IntSize()} }

/*
 * Float
 */

type remoteFloat struct {
	r    remote
	size int
}

func (v remoteFloat) String() string {
	return tryRVString(func(a aborter) string { return fmt.Sprintf("%v", v.aGet(a)) })
}

func (v remoteFloat) Assign(t *eval.Thread, o eval.Value) {
	v.Set(t, o.(eval.FloatValue).Get(t))
}

func (v remoteFloat) Get(t *eval.Thread) float64 {
	return v.aGet(t)
}

func (v remoteFloat) aGet(a aborter) float64 {
	bits := v.r.Get(a, v.size)
	switch v.size {
	case 4:
		return float64(v.r.p.ToFloat32(uint32(bits)))
	case 8:
		return v.r.p.ToFloat64(bits)
	}
	panic("Unexpected float size")
}

func (v remoteFloat) Set(t *eval.Thread, x float64) {
	v.aSet(t, x)
}

func (v remoteFloat) aSet(a aborter, x float64) {
	var bits uint64
	switch v.size {
	case 4:
		bits = uint64(v.r.p.FromFloat32(float32(x)))
	case 8:
		bits = v.r.p.FromFloat64(x)
	default:
		panic("Unexpected float size")
	}
	v.r.Set(a, v.size, bits)
}

func (v remoteFloat) addr() remote { return v.r }

func mkFloat32(r remote) eval.Value { return remoteFloat{r, 4} }

func mkFloat64(r remote) eval.Value { return remoteFloat{r, 8} }

func mkFloat(r remote) eval.Value { return remoteFloat{r, r.p.FloatSize()} }

/*
 * String
 */

type remoteString struct {
	r remote
}

func (v remoteString) String() string {
	return tryRVString(func(a aborter) string { return v.aGet(a) })
}

func (v remoteString) Assign(t *eval.Thread, o eval.Value) {
	v.Set(t, o.(eval.StringValue).Get(t))
}

func (v remoteString) Get(t *eval.Thread) string {
	return v.aGet(t)
}

func (v remoteString) aGet(a aborter) string {
	rs := v.r.p.runtime.String.mk(v.r).(remoteStruct)
	str := proc.Word(rs.field(v.r.p.f.String.Str).(remoteUint).aGet(a))
	len := rs.field(v.r.p.f.String.Len).(remoteInt).aGet(a)

	bytes := make([]uint8, len)
	_, err := v.r.p.Peek(str, bytes)
	if err != nil {
		a.Abort(err)
	}
	return string(bytes)
}

func (v remoteString) Set(t *eval.Thread, x string) {
	v.aSet(t, x)
}

func (v remoteString) aSet(a aborter, x string) {
	// TODO(austin) This isn't generally possible without the
	// ability to allocate remote memory.
	a.Abort(ReadOnlyError("remote strings cannot be assigned to"))
}

func mkString(r remote) eval.Value { return remoteString{r} }

/*
 * Array
 */

type remoteArray struct {
	r        remote
	len      int64
	elemType *remoteType
}

func (v remoteArray) String() string {
	res := "{"
	for i := int64(0); i < v.len; i++ {
		if i > 0 {
			res += ", "
		}
		res += v.elem(i).String()
	}
	return res + "}"
}

func (v remoteArray) Assign(t *eval.Thread, o eval.Value) {
	// TODO(austin) Could do a bigger memcpy if o is a
	// remoteArray in the same Process.
	oa := o.(eval.ArrayValue)
	for i := int64(0); i < v.len; i++ {
		v.Elem(t, i).Assign(t, oa.Elem(t, i))
	}
}

func (v remoteArray) Get(t *eval.Thread) eval.ArrayValue {
	return v
}

func (v remoteArray) Elem(t *eval.Thread, i int64) eval.Value {
	return v.elem(i)
}

func (v remoteArray) elem(i int64) eval.Value {
	return v.elemType.mk(v.r.plus(proc.Word(int64(v.elemType.size) * i)))
}

func (v remoteArray) Sub(i int64, len int64) eval.ArrayValue {
	return remoteArray{v.r.plus(proc.Word(int64(v.elemType.size) * i)), len, v.elemType}
}

/*
 * Struct
 */

type remoteStruct struct {
	r      remote
	layout []remoteStructField
}

type remoteStructField struct {
	offset    int
	fieldType *remoteType
}

func (v remoteStruct) String() string {
	res := "{"
	for i := range v.layout {
		if i > 0 {
			res += ", "
		}
		res += v.field(i).String()
	}
	return res + "}"
}

func (v remoteStruct) Assign(t *eval.Thread, o eval.Value) {
	// TODO(austin) Could do a bigger memcpy.
	oa := o.(eval.StructValue)
	l := len(v.layout)
	for i := 0; i < l; i++ {
		v.Field(t, i).Assign(t, oa.Field(t, i))
	}
}

func (v remoteStruct) Get(t *eval.Thread) eval.StructValue {
	return v
}

func (v remoteStruct) Field(t *eval.Thread, i int) eval.Value {
	return v.field(i)
}

func (v remoteStruct) field(i int) eval.Value {
	f := &v.layout[i]
	return f.fieldType.mk(v.r.plus(proc.Word(f.offset)))
}

func (v remoteStruct) addr() remote { return v.r }

/*
 * Pointer
 */

// TODO(austin) Comparing two remote pointers for equality in the
// interpreter will crash it because the Value's returned from
// remotePtr.Get() will be structs.

type remotePtr struct {
	r        remote
	elemType *remoteType
}

func (v remotePtr) String() string {
	return tryRVString(func(a aborter) string {
		e := v.aGet(a)
		if e == nil {
			return "<nil>"
		}
		return "&" + e.String()
	})
}

func (v remotePtr) Assign(t *eval.Thread, o eval.Value) {
	v.Set(t, o.(eval.PtrValue).Get(t))
}

func (v remotePtr) Get(t *eval.Thread) eval.Value {
	return v.aGet(t)
}

func (v remotePtr) aGet(a aborter) eval.Value {
	addr := proc.Word(v.r.Get(a, v.r.p.PtrSize()))
	if addr == 0 {
		return nil
	}
	return v.elemType.mk(remote{addr, v.r.p})
}

func (v remotePtr) Set(t *eval.Thread, x eval.Value) {
	v.aSet(t, x)
}

func (v remotePtr) aSet(a aborter, x eval.Value) {
	if x == nil {
		v.r.Set(a, v.r.p.PtrSize(), 0)
		return
	}
	xr, ok := x.(remoteValue)
	if !ok || v.r.p != xr.addr().p {
		a.Abort(RemoteMismatchError("remote pointer must point within the same process"))
	}
	v.r.Set(a, v.r.p.PtrSize(), uint64(xr.addr().base))
}

func (v remotePtr) addr() remote { return v.r }

/*
 * Slice
 */

type remoteSlice struct {
	r        remote
	elemType *remoteType
}

func (v remoteSlice) String() string {
	return tryRVString(func(a aborter) string {
		b := v.aGet(a).Base
		if b == nil {
			return "<nil>"
		}
		return b.String()
	})
}

func (v remoteSlice) Assign(t *eval.Thread, o eval.Value) {
	v.Set(t, o.(eval.SliceValue).Get(t))
}

func (v remoteSlice) Get(t *eval.Thread) eval.Slice {
	return v.aGet(t)
}

func (v remoteSlice) aGet(a aborter) eval.Slice {
	rs := v.r.p.runtime.Slice.mk(v.r).(remoteStruct)
	base := proc.Word(rs.field(v.r.p.f.Slice.Array).(remoteUint).aGet(a))
	nel := rs.field(v.r.p.f.Slice.Len).(remoteInt).aGet(a)
	cap := rs.field(v.r.p.f.Slice.Cap).(remoteInt).aGet(a)
	if base == 0 {
		return eval.Slice{nil, nel, cap}
	}
	return eval.Slice{remoteArray{remote{base, v.r.p}, nel, v.elemType}, nel, cap}
}

func (v remoteSlice) Set(t *eval.Thread, x eval.Slice) {
	v.aSet(t, x)
}

func (v remoteSlice) aSet(a aborter, x eval.Slice) {
	rs := v.r.p.runtime.Slice.mk(v.r).(remoteStruct)
	if x.Base == nil {
		rs.field(v.r.p.f.Slice.Array).(remoteUint).aSet(a, 0)
	} else {
		ar, ok := x.Base.(remoteArray)
		if !ok || v.r.p != ar.r.p {
			a.Abort(RemoteMismatchError("remote slice must point within the same process"))
		}
		rs.field(v.r.p.f.Slice.Array).(remoteUint).aSet(a, uint64(ar.r.base))
	}
	rs.field(v.r.p.f.Slice.Len).(remoteInt).aSet(a, x.Len)
	rs.field(v.r.p.f.Slice.Cap).(remoteInt).aSet(a, x.Cap)
}
