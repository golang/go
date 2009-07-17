// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

// TODO(rsc): When garbage collector changes, revisit
// the allocations in this file that use unsafe.Pointer.

import (
	"bytes";
	"gob";
	"io";
	"math";
	"os";
	"reflect";
	"unsafe";
)

// The global execution state of an instance of the decoder.
type decodeState struct {
	b	*bytes.Buffer;
	err	os.Error;
	fieldnum	int;	// the last field number read.
}

// decodeUintReader reads an encoded unsigned integer from an io.Reader.
// Used only by the Decoder to read the message length.
func decodeUintReader(r io.Reader, oneByte []byte) (x uint64, err os.Error) {
	for shift := uint(0);; shift += 7 {
		var n int;
		n, err = r.Read(oneByte);
		if err != nil {
			return 0, err
		}
		b := oneByte[0];
		x |= uint64(b) << shift;
		if b&0x80 != 0 {
			x &^= 0x80 << shift;
			break
		}
	}
	return x, nil;
}

// decodeUint reads an encoded unsigned integer from state.r.
// Sets state.err.  If state.err is already non-nil, it does nothing.
func decodeUint(state *decodeState) (x uint64) {
	if state.err != nil {
		return
	}
	for shift := uint(0);; shift += 7 {
		var b uint8;
		b, state.err = state.b.ReadByte();
		if state.err != nil {
			return 0
		}
		x |= uint64(b) << shift;
		if b&0x80 != 0 {
			x &^= 0x80 << shift;
			break
		}
	}
	return x;
}

// decodeInt reads an encoded signed integer from state.r.
// Sets state.err.  If state.err is already non-nil, it does nothing.
func decodeInt(state *decodeState) int64 {
	x := decodeUint(state);
	if state.err != nil {
		return 0
	}
	if x & 1 != 0 {
		return ^int64(x>>1)
	}
	return int64(x >> 1)
}

type decInstr struct
type decOp func(i *decInstr, state *decodeState, p unsafe.Pointer);

// The 'instructions' of the decoding machine
type decInstr struct {
	op	decOp;
	field		int;	// field number of the wire type
	indir	int;	// how many pointer indirections to reach the value in the struct
	offset	uintptr;	// offset in the structure of the field to encode
}

// Since the encoder writes no zeros, if we arrive at a decoder we have
// a value to extract and store.  The field number has already been read
// (it's how we knew to call this decoder).
// Each decoder is responsible for handling any indirections associated
// with the data structure.  If any pointer so reached is nil, allocation must
// be done.

// Walk the pointer hierarchy, allocating if we find a nil.  Stop one before the end.
func decIndirect(p unsafe.Pointer, indir int) unsafe.Pointer {
	for ; indir > 1; indir-- {
		if *(*unsafe.Pointer)(p) == nil {
			// Allocation required
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(unsafe.Pointer));
		}
		p = *(*unsafe.Pointer)(p);
	}
	return p
}

func ignoreUint(i *decInstr, state *decodeState, p unsafe.Pointer) {
	decodeUint(state);
}

func decBool(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(bool));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*bool)(p) = decodeInt(state) != 0;
}

func decInt(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int)(p) = int(decodeInt(state));
}

func decUint(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint)(p) = uint(decodeUint(state));
}

func decInt8(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int8));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int8)(p) = int8(decodeInt(state));
}

func decUint8(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint8));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint8)(p) = uint8(decodeUint(state));
}

func decInt16(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int16));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int16)(p) = int16(decodeInt(state));
}

func decUint16(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint16));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint16)(p) = uint16(decodeUint(state));
}

func decInt32(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int32));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int32)(p) = int32(decodeInt(state));
}

func decUint32(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint32));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint32)(p) = uint32(decodeUint(state));
}

func decInt64(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int64));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int64)(p) = int64(decodeInt(state));
}

func decUint64(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint64));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint64)(p) = uint64(decodeUint(state));
}

func decUintptr(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uintptr));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uintptr)(p) = uintptr(decodeUint(state));
}

// Floating-point numbers are transmitted as uint64s holding the bits
// of the underlying representation.  They are sent byte-reversed, with
// the exponent end coming out first, so integer floating point numbers
// (for example) transmit more compactly.  This routine does the
// unswizzling.
func floatFromBits(u uint64) float64 {
	var v uint64;
	for i := 0; i < 8; i++ {
		v <<= 8;
		v |= u & 0xFF;
		u >>= 8;
	}
	return math.Float64frombits(v);
}

func decFloat(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*float)(p) = float(floatFromBits(uint64(decodeUint(state))));
}

func decFloat32(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float32));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*float32)(p) = float32(floatFromBits(uint64(decodeUint(state))));
}

func decFloat64(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float64));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*float64)(p) = floatFromBits(uint64(decodeUint(state)));
}

// uint8 arrays are encoded as an unsigned count followed by the raw bytes.
func decUint8Array(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new([]uint8));
		}
		p = *(*unsafe.Pointer)(p);
	}
	b := make([]uint8, decodeUint(state));
	state.b.Read(b);
	*(*[]uint8)(p) = b;
}

// Strings are encoded as an unsigned count followed by the raw bytes.
func decString(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new([]byte));
		}
		p = *(*unsafe.Pointer)(p);
	}
	b := make([]byte, decodeUint(state));
	state.b.Read(b);
	*(*string)(p) = string(b);
}

func ignoreUint8Array(i *decInstr, state *decodeState, p unsafe.Pointer) {
	b := make([]byte, decodeUint(state));
	state.b.Read(b);
}

// Execution engine

// The encoder engine is an array of instructions indexed by field number of the incoming
// data.  It is executed with random access according to field number.
type decEngine struct {
	instr	[]decInstr;
	numInstr	int;	// the number of active instructions
}

func decodeStruct(engine *decEngine, rtyp *reflect.StructType, b *bytes.Buffer, p uintptr, indir int) os.Error {
	if indir > 0 {
		up := unsafe.Pointer(p);
		if *(*unsafe.Pointer)(up) == nil {
			// Allocate the structure by making a slice of bytes and recording the
			// address of the beginning of the array. TODO(rsc).
			b := make([]byte, rtyp.Size());
			*(*unsafe.Pointer)(up) = unsafe.Pointer(&b[0]);
		}
		p = *(*uintptr)(up);
	}
	state := new(decodeState);
	state.b = b;
	state.fieldnum = -1;
	basep := p;
	for state.err == nil {
		delta := int(decodeUint(state));
		if delta < 0 {
			state.err = os.ErrorString("gob decode: corrupted data: negative delta");
			break
		}
		if state.err != nil || delta == 0 {	// struct terminator is zero delta fieldnum
			break
		}
		fieldnum := state.fieldnum + delta;
		if fieldnum >= len(engine.instr) {
			panicln("TODO(r): field number out of range", fieldnum, len(engine.instr));
		}
		instr := &engine.instr[fieldnum];
		p := unsafe.Pointer(basep+instr.offset);
		if instr.indir > 1 {
			p = decIndirect(p, instr.indir);
		}
		instr.op(instr, state, p);
		state.fieldnum = fieldnum;
	}
	return state.err
}

func decodeArrayHelper(state *decodeState, p uintptr, elemOp decOp, elemWid uintptr, length, elemIndir int) os.Error {
	instr := &decInstr{elemOp, 0, elemIndir, 0};
	for i := 0; i < length && state.err == nil; i++ {
		up := unsafe.Pointer(p);
		if elemIndir > 1 {
			up = decIndirect(up, elemIndir);
		}
		elemOp(instr, state, up);
		p += uintptr(elemWid);
	}
	return state.err
}

func decodeArray(atyp *reflect.ArrayType, state *decodeState, p uintptr, elemOp decOp, elemWid uintptr, length, indir, elemIndir int) os.Error {
	if indir > 0 {
		up := unsafe.Pointer(p);
		if *(*unsafe.Pointer)(up) == nil {
			// Allocate the array by making a slice of bytes of the correct size
			// and taking the address of the beginning of the array. TODO(rsc).
			b := make([]byte, atyp.Size());
			*(**byte)(up) = &b[0];
		}
		p = *(*uintptr)(up);
	}
	if n := decodeUint(state); n != uint64(length) {
		return os.ErrorString("gob: length mismatch in decodeArray");
	}
	return decodeArrayHelper(state, p, elemOp, elemWid, length, elemIndir);
}

func ignoreArrayHelper(state *decodeState, elemOp decOp, length int) os.Error {
	instr := &decInstr{elemOp, 0, 0, 0};
	for i := 0; i < length && state.err == nil; i++ {
		elemOp(instr, state, nil);
	}
	return state.err
}

func ignoreArray(state *decodeState, elemOp decOp, length int) os.Error {
	if n := decodeUint(state); n != uint64(length) {
		return os.ErrorString("gob: length mismatch in ignoreArray");
	}
	return ignoreArrayHelper(state, elemOp, length);
}

func decodeSlice(atyp *reflect.SliceType, state *decodeState, p uintptr, elemOp decOp, elemWid uintptr, indir, elemIndir int) os.Error {
	length := uintptr(decodeUint(state));
	if indir > 0 {
		up := unsafe.Pointer(p);
		if *(*unsafe.Pointer)(up) == nil {
			// Allocate the slice header.
			*(*unsafe.Pointer)(up) = unsafe.Pointer(new(reflect.SliceHeader));
		}
		p = *(*uintptr)(up);
	}
	// Allocate storage for the slice elements, that is, the underlying array.
	data := make([]byte, length*atyp.Elem().Size());
	// Always write a header at p.
	hdrp := (*reflect.SliceHeader)(unsafe.Pointer(p));
	hdrp.Data = uintptr(unsafe.Pointer(&data[0]));
	hdrp.Len = uint32(length);
	hdrp.Cap = uint32(length);
	return decodeArrayHelper(state, hdrp.Data, elemOp, elemWid, int(length), elemIndir);
}

func ignoreSlice(state *decodeState, elemOp decOp) os.Error {
	return ignoreArrayHelper(state, elemOp, int(decodeUint(state)));
}

var decOpMap = map[reflect.Type] decOp {
	reflect.Typeof((*reflect.BoolType)(nil)): decBool,
	reflect.Typeof((*reflect.IntType)(nil)): decInt,
	reflect.Typeof((*reflect.Int8Type)(nil)): decInt8,
	reflect.Typeof((*reflect.Int16Type)(nil)): decInt16,
	reflect.Typeof((*reflect.Int32Type)(nil)): decInt32,
	reflect.Typeof((*reflect.Int64Type)(nil)): decInt64,
	reflect.Typeof((*reflect.UintType)(nil)): decUint,
	reflect.Typeof((*reflect.Uint8Type)(nil)): decUint8,
	reflect.Typeof((*reflect.Uint16Type)(nil)): decUint16,
	reflect.Typeof((*reflect.Uint32Type)(nil)): decUint32,
	reflect.Typeof((*reflect.Uint64Type)(nil)): decUint64,
	reflect.Typeof((*reflect.UintptrType)(nil)): decUintptr,
	reflect.Typeof((*reflect.FloatType)(nil)): decFloat,
	reflect.Typeof((*reflect.Float32Type)(nil)): decFloat32,
	reflect.Typeof((*reflect.Float64Type)(nil)): decFloat64,
	reflect.Typeof((*reflect.StringType)(nil)): decString,
}

var decIgnoreOpMap = map[TypeId] decOp {
	tBool: ignoreUint,
	tInt: ignoreUint,
	tUint: ignoreUint,
	tFloat: ignoreUint,
	tBytes: ignoreUint8Array,
	tString: ignoreUint8Array,
}

func getDecEnginePtr(wireId TypeId, rt reflect.Type) **decEngine

// Return the decoding op for the base type under rt and
// the indirection count to reach it.
func decOpFor(wireId TypeId, rt reflect.Type) (decOp, int) {
	typ, indir := indirect(rt);
	op, ok := decOpMap[reflect.Typeof(typ)];
	if !ok {
		// Special cases
		switch t := typ.(type) {
		case *reflect.SliceType:
			if _, ok := t.Elem().(*reflect.Uint8Type); ok {
				op = decUint8Array;
				break;
			}
			elemId := wireId.gobType().(*sliceType).Elem;
			elemOp, elemIndir := decOpFor(elemId, t.Elem());
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.err = decodeSlice(t, state, uintptr(p), elemOp, t.Elem().Size(), i.indir, elemIndir);
			};

		case *reflect.ArrayType:
			elemId := wireId.gobType().(*arrayType).Elem;
			elemOp, elemIndir := decOpFor(elemId, t.Elem());
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.err = decodeArray(t, state, uintptr(p), elemOp, t.Elem().Size(), t.Len(), i.indir, elemIndir);
			};

		case *reflect.StructType:
			// Generate a closure that calls out to the engine for the nested type.
			enginePtr := getDecEnginePtr(wireId, typ);
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				// indirect through info to delay evaluation for recursive structs
				state.err = decodeStruct(*enginePtr, t, state.b, uintptr(p), i.indir)
			};
		}
	}
	if op == nil {
		panicln("decode can't handle type", rt.String());
	}
	return op, indir
}

// Return the decoding op for a field that has no destination.
func decIgnoreOpFor(wireId TypeId) decOp {
	op, ok := decIgnoreOpMap[wireId];
	if !ok {
		// Special cases
		switch t := wireId.gobType().(type) {
		case *sliceType:
			elemId := wireId.gobType().(*sliceType).Elem;
			elemOp := decIgnoreOpFor(elemId);
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.err = ignoreSlice(state, elemOp);
			};

		case *arrayType:
			elemId := wireId.gobType().(*arrayType).Elem;
			elemOp := decIgnoreOpFor(elemId);
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.err = ignoreArray(state, elemOp, t.Len);
			};

		case *structType:
			// TODO: write an ignore engine for structs
		}
	}
	if op == nil {
		panicln("decode can't handle type", wireId.gobType().String());
	}
	return op;
}

// Are these two gob Types compatible?
// Answers the question for basic types, arrays, and slices.  Defers for structs.
func compatibleType(fr reflect.Type, fw TypeId) bool {
	for {
		if pt, ok := fr.(*reflect.PtrType); ok {
			fr = pt.Elem();
			continue;
		}
		break;
	}
	switch t := fr.(type) {
	default:
		// interface, map, chan, etc: cannot handle.
		return false;
	case *reflect.BoolType:
		return fw == tBool;
	case *reflect.IntType:
		return fw == tInt;
	case *reflect.Int8Type:
		return fw == tInt;
	case *reflect.Int16Type:
		return fw == tInt;
	case *reflect.Int32Type:
		return fw == tInt;
	case *reflect.Int64Type:
		return fw == tInt;
	case *reflect.UintType:
		return fw == tUint;
	case *reflect.Uint8Type:
		return fw == tUint;
	case *reflect.Uint16Type:
		return fw == tUint;
	case *reflect.Uint32Type:
		return fw == tUint;
	case *reflect.Uint64Type:
		return fw == tUint;
	case *reflect.UintptrType:
		return fw == tUint;
	case *reflect.FloatType:
		return fw == tFloat;
	case *reflect.Float32Type:
		return fw == tFloat;
	case *reflect.Float64Type:
		return fw == tFloat;
	case *reflect.StringType:
		return fw == tString;
	case *reflect.StructType:
		return true;	// defer for now
	case *reflect.ArrayType:
		aw, ok := fw.gobType().(*arrayType);
		return ok && t.Len() == aw.Len && compatibleType(t.Elem(), aw.Elem);
	case *reflect.SliceType:
		// Is it an array of bytes?
		et := t.Elem();
		if _, ok := et.(*reflect.Uint8Type); ok {
			return fw == tBytes
		}
		sw, ok := fw.gobType().(*sliceType);
		return ok && compatibleType(t.Elem(), sw.Elem);
	}
	return true;
}

func compileDec(wireId TypeId, rt reflect.Type) *decEngine {
	srt, ok1 := rt.(*reflect.StructType);
	wireStruct, ok2 := wireId.gobType().(*structType);
	if !ok1 || !ok2 {
		panicln("gob: TODO: can't handle non-structs");
	}
	engine := new(decEngine);
	engine.instr = make([]decInstr, len(wireStruct.field));
	// Loop over the fields of the wire type.
	for fieldnum := 0; fieldnum < len(wireStruct.field); fieldnum++ {
		wireField := wireStruct.field[fieldnum];
		// Find the field of the local type with the same name.
		// TODO: put this as a method in reflect
		var localField reflect.StructField;
		for lfn := 0; lfn < srt.NumField(); lfn++ {
			if srt.Field(lfn).Name == wireField.name {
				localField = srt.Field(lfn);
				break;
			}
		}
		// TODO(r): anonymous names
		if localField.Anonymous || localField.Name == "" {
			println("no matching field", wireField.name, "in type", wireId.String());
			op := decIgnoreOpFor(wireField.typeId);
			engine.instr[fieldnum] = decInstr{op, fieldnum, 0, 0};
			continue;
		}
		if !compatibleType(localField.Type, wireField.typeId) {
			panicln("TODO: wrong type for field", wireField.name, "in type", wireId.String());
		}
		op, indir := decOpFor(wireField.typeId, localField.Type);
		engine.instr[fieldnum] = decInstr{op, fieldnum, indir, uintptr(localField.Offset)};
		engine.numInstr++;
	}
	return engine;
}


// typeLock must be held.
func getDecEnginePtr(wireId TypeId, rt reflect.Type) **decEngine {
	info := getTypeInfo(rt);	// TODO: eliminate this; creates a gobType you don't need.
	var enginePtr **decEngine;
	var ok bool;
	if enginePtr, ok = info.decoderPtr[wireId]; !ok {
		if info.typeId.gobType() == nil {
			_pkg, name := rt.Name();
			info.typeId = newType(name, rt).id();
		}
		// mark this engine as underway before compiling to handle recursive types.
		enginePtr = new(*decEngine);
		info.decoderPtr[wireId] = enginePtr;
		*enginePtr = compileDec(wireId, rt);
	}
	return enginePtr
}

func decode(b *bytes.Buffer, wireId TypeId, e interface{}) os.Error {
	// Dereference down to the underlying object.
	rt, indir := indirect(reflect.Typeof(e));
	v := reflect.NewValue(e);
	for i := 0; i < indir; i++ {
		v = reflect.Indirect(v);
	}
	var st *reflect.StructValue;
	var ok bool;
	if st, ok = v.(*reflect.StructValue); !ok {
		return os.ErrorString("gob: decode can't handle " + rt.String())
	}
	typeLock.Lock();
	engine := *getDecEnginePtr(wireId, rt);
	typeLock.Unlock();
	if engine.numInstr == 0 && st.NumField() > 0 {
		path, name := rt.Name();
		return os.ErrorString("no fields matched compiling decoder for " + name)
	}
	return decodeStruct(engine, rt.(*reflect.StructType), b, uintptr(v.Addr()), 0);
}
