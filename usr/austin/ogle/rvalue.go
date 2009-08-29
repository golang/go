// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"eval";
	"fmt";
	"ptrace";
)

// A RemoteMismatchError occurs when an operation that requires two
// identical remote processes is given different process.  For
// example, this occurs when trying to set a pointer in one process to
// point to something in another process.
type RemoteMismatchError string

func (e RemoteMismatchError) String() string {
	return string(e);
}

// A maker is a function that converts a remote address into an
// interpreter Value.
type maker func(remote) eval.Value

type remoteValue interface {
	addr() remote;
}

// remote represents an address in a remote process.
type remote struct {
	base ptrace.Word;
	p *Process;
}

func (v remote) Get(size int) uint64 {
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
	var arr [8]byte;
	buf := arr[0:size];
	_, err := v.p.thread.Peek(v.base, buf);
	if err != nil {
		eval.Abort(err);
	}
	return uint64(v.p.ToWord(buf));
}

func (v remote) Set(size int, x uint64) {
	var arr [8]byte;
	buf := arr[0:size];
	v.p.FromWord(ptrace.Word(x), buf);
	_, err := v.p.thread.Poke(v.base, buf);
	if err != nil {
		eval.Abort(err);
	}
}

func (v remote) plus(x ptrace.Word) remote {
	return remote{v.base + x, v.p};
}

/*
 * Bool
 */

type remoteBool struct {
	r remote;
}

func (v remoteBool) String() string {
	return fmt.Sprintf("%v", v.Get());
}

func (v remoteBool) Assign(o eval.Value) {
	v.Set(o.(eval.BoolValue).Get());
}

func (v remoteBool) Get() bool {
	return v.r.Get(1) != 0;
}

func (v remoteBool) Set(x bool) {
	if x {
		v.r.Set(1, 1);
	} else {
		v.r.Set(1, 0);
	}
}

func (v remoteBool) addr() remote {
	return v.r;
}

func mkBool(r remote) eval.Value {
	return remoteBool{r};
}

/*
 * Uint
 */

type remoteUint struct {
	r remote;
	size int;
}

func (v remoteUint) String() string {
	return fmt.Sprintf("%v", v.Get());
}

func (v remoteUint) Assign(o eval.Value) {
	v.Set(o.(eval.UintValue).Get());
}

func (v remoteUint) Get() uint64 {
	return v.r.Get(v.size);
}

func (v remoteUint) Set(x uint64) {
	v.r.Set(v.size, x);
}

func (v remoteUint) addr() remote {
	return v.r;
}

func mkUint8(r remote) eval.Value {
	return remoteUint{r, 1};
}

func mkUint16(r remote) eval.Value {
	return remoteUint{r, 2};
}

func mkUint32(r remote) eval.Value {
	return remoteUint{r, 4};
}

func mkUint64(r remote) eval.Value {
	return remoteUint{r, 8};
}

func mkUint(r remote) eval.Value {
	return remoteUint{r, r.p.IntSize()};
}

func mkUintptr(r remote) eval.Value {
	return remoteUint{r, r.p.PtrSize()};
}

/*
 * Int
 */

type remoteInt struct {
	r remote;
	size int;
}

func (v remoteInt) String() string {
	return fmt.Sprintf("%v", v.Get());
}

func (v remoteInt) Assign(o eval.Value) {
	v.Set(o.(eval.IntValue).Get());
}

func (v remoteInt) Get() int64 {
	return int64(v.r.Get(v.size));
}

func (v remoteInt) Set(x int64) {
	v.r.Set(v.size, uint64(x));
}

func (v remoteInt) addr() remote {
	return v.r;
}

func mkInt8(r remote) eval.Value {
	return remoteInt{r, 1};
}

func mkInt16(r remote) eval.Value {
	return remoteInt{r, 2};
}

func mkInt32(r remote) eval.Value {
	return remoteInt{r, 4};
}

func mkInt64(r remote) eval.Value {
	return remoteInt{r, 8};
}

func mkInt(r remote) eval.Value {
	return remoteInt{r, r.p.IntSize()};
}

/*
 * Float
 */

type remoteFloat struct {
	r remote;
	size int;
}

func (v remoteFloat) String() string {
	return fmt.Sprintf("%v", v.Get());
}

func (v remoteFloat) Assign(o eval.Value) {
	v.Set(o.(eval.FloatValue).Get());
}

func (v remoteFloat) Get() float64 {
	bits := v.r.Get(v.size);
	switch v.size {
	case 4:
		return float64(v.r.p.ToFloat32(uint32(bits)));
	case 8:
		return v.r.p.ToFloat64(bits);
	}
	panic("Unexpected float size ", v.size);
}

func (v remoteFloat) Set(x float64) {
	var bits uint64;
	switch v.size{
	case 4:
		bits = uint64(v.r.p.FromFloat32(float32(x)));
	case 8:
		bits = v.r.p.FromFloat64(x);
	default:
		panic("Unexpected float size ", v.size);
	}
	v.r.Set(v.size, bits);
}

func (v remoteFloat) addr() remote {
	return v.r;
}

func mkFloat32(r remote) eval.Value {
	return remoteFloat{r, 4};
}

func mkFloat64(r remote) eval.Value {
	return remoteFloat{r, 8};
}

func mkFloat(r remote) eval.Value {
	return remoteFloat{r, r.p.FloatSize()};
}

/*
 * String
 */

type remoteString struct {
	r remote;
}

func (v remoteString) String() string {
	return v.Get();
}

func (v remoteString) Assign(o eval.Value) {
	v.Set(o.(eval.StringValue).Get());
}

func (v remoteString) Get() string {
	rs := v.r.p.runtime.String.mk(v.r).(remoteStruct);
	str := ptrace.Word(rs.Field(v.r.p.f.String.Str).(remoteUint).Get());
	len := rs.Field(v.r.p.f.String.Len).(remoteInt).Get();
	
	bytes := make([]uint8, len);
	_, err := v.r.p.thread.Peek(str, bytes);
	if err != nil {
		eval.Abort(err);
	}
	return string(bytes);
}

func (v remoteString) Set(x string) {
	// TODO(austin) This isn't generally possible without the
	// ability to allocate remote memory.
	eval.Abort(RemoteMismatchError("remote strings cannot be assigned to"));
}

func mkString(r remote) eval.Value {
	return remoteString{r};
}

/*
 * Array
 */

type remoteArray struct {
	r remote;
	len int64;
	elemType *remoteType;
}

func (v remoteArray) String() string {
	res := "{";
	for i := int64(0); i < v.len; i++ {
		if i > 0 {
			res += ", ";
		}
		res += v.Elem(i).String();
	}
	return res + "}";
}

func (v remoteArray) Assign(o eval.Value) {
 	// TODO(austin) Could do a bigger memcpy if o is a
	// remoteArray in the same Process.
	oa := o.(eval.ArrayValue);
	for i := int64(0); i < v.len; i++ {
		v.Elem(i).Assign(oa.Elem(i));
	}
}

func (v remoteArray) Get() eval.ArrayValue {
	return v;
}

func (v remoteArray) Elem(i int64) eval.Value {
	return v.elemType.mk(v.r.plus(ptrace.Word(int64(v.elemType.size) * i)));
}

func (v remoteArray) From(i int64) eval.ArrayValue {
	return remoteArray{v.r.plus(ptrace.Word(int64(v.elemType.size) * i)), v.len - i, v.elemType};
}

/*
 * Struct
 */

type remoteStruct struct {
	r remote;
	layout []remoteStructField;
}

type remoteStructField struct {
	offset int;
	fieldType *remoteType;
}

func (v remoteStruct) String() string {
	res := "{";
	for i := range v.layout {
		if i > 0 {
			res += ", ";
		}
		res += v.Field(i).String();
	}
	return res + "}";
}

func (v remoteStruct) Assign(o eval.Value) {
	// TODO(austin) Could do a bigger memcpy.
	oa := o.(eval.StructValue);
	l := len(v.layout);
	for i := 0; i < l; i++ {
		v.Field(i).Assign(oa.Field(i));
	}
}

func (v remoteStruct) Get() eval.StructValue {
	return v;
}

func (v remoteStruct) Field(i int) eval.Value {
	f := &v.layout[i];
	return f.fieldType.mk(v.r.plus(ptrace.Word(f.offset)));
}

func (v remoteStruct) addr() remote {
	return v.r;
}

/*
 * Pointer
 */

// TODO(austin) Comparing two remote pointers for equality in the
// interpreter will crash it because the Value's returned from
// remotePtr.Get() will be structs.

type remotePtr struct {
	r remote;
	elemType *remoteType;
}

func (v remotePtr) String() string {
	e := v.Get();
	if e == nil {
		return "<nil>";
	}
	return "&" + e.String();
}

func (v remotePtr) Assign(o eval.Value) {
	v.Set(o.(eval.PtrValue).Get());
}

func (v remotePtr) Get() eval.Value {
	addr := ptrace.Word(v.r.Get(v.r.p.PtrSize()));
	if addr == 0 {
		return nil;
	}
	return v.elemType.mk(remote{addr, v.r.p});
}

func (v remotePtr) Set(x eval.Value) {
	if x == nil {
		v.r.Set(v.r.p.PtrSize(), 0);
		return;
	}
	xr, ok := x.(remoteValue);
	if !ok || v.r.p != xr.addr().p {
		eval.Abort(RemoteMismatchError("remote pointer must point within the same process"));
	}
	v.r.Set(v.r.p.PtrSize(), uint64(xr.addr().base));
}

func (v remotePtr) addr() remote {
	return v.r;
}

/*
 * Slice
 */

type remoteSlice struct {
	r remote;
	elemType *remoteType;
}

func (v remoteSlice) String() string {
	b := v.Get().Base;
	if b == nil {
		return "<nil>";
	}
	return b.String();
}

func (v remoteSlice) Assign(o eval.Value) {
	v.Set(o.(eval.SliceValue).Get());
}

func (v remoteSlice) Get() eval.Slice {
	rs := v.r.p.runtime.Slice.mk(v.r).(remoteStruct);
	base := ptrace.Word(rs.Field(v.r.p.f.Slice.Array).(remoteUint).Get());
	nel := rs.Field(v.r.p.f.Slice.Len).(remoteInt).Get();
	cap := rs.Field(v.r.p.f.Slice.Cap).(remoteInt).Get();
	if base == 0 {
		return eval.Slice{nil, nel, cap};
	}
	return eval.Slice{remoteArray{remote{base, v.r.p}, nel, v.elemType}, nel, cap};
}

func (v remoteSlice) Set(x eval.Slice) {
	rs := v.r.p.runtime.Slice.mk(v.r).(remoteStruct);
	if x.Base == nil {
		rs.Field(v.r.p.f.Slice.Array).(remoteUint).Set(0);
	} else {
		ar, ok := x.Base.(remoteArray);
		if !ok || v.r.p != ar.r.p {
			eval.Abort(RemoteMismatchError("remote slice must point within the same process"));
		}
		rs.Field(v.r.p.f.Slice.Array).(remoteUint).Set(uint64(ar.r.base));
	}
	rs.Field(v.r.p.f.Slice.Len).(remoteInt).Set(x.Len);
	rs.Field(v.r.p.f.Slice.Cap).(remoteInt).Set(x.Cap);
}
