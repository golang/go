// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

// TODO(rsc): When garbage collector changes, revisit
// the allocations in this file that use unsafe.Pointer.

import (
	"gob";
	"io";
	"math";
	"os";
	"reflect";
	"unsafe";
)

// The global execution state of an instance of the decoder.
type DecState struct {
	r	io.Reader;
	err	os.Error;
	fieldnum	int;	// the last field number read.
	buf [1]byte;	// buffer used by the decoder; here to avoid allocation.
}

// DecodeUint reads an encoded unsigned integer from state.r.
// Sets state.err.  If state.err is already non-nil, it does nothing.
func DecodeUint(state *DecState) (x uint64) {
	if state.err != nil {
		return
	}
	for shift := uint(0);; shift += 7 {
		var n int;
		n, state.err = state.r.Read(&state.buf);
		if n != 1 {
			return 0
		}
		b := uint64(state.buf[0]);
		x |= b << shift;
		if b&0x80 != 0 {
			x &^= 0x80 << shift;
			break
		}
	}
	return x;
}

// DecodeInt reads an encoded signed integer from state.r.
// Sets state.err.  If state.err is already non-nil, it does nothing.
func DecodeInt(state *DecState) int64 {
	x := DecodeUint(state);
	if state.err != nil {
		return 0
	}
	if x & 1 != 0 {
		return ^int64(x>>1)
	}
	return int64(x >> 1)
}

type decInstr struct
type decOp func(i *decInstr, state *DecState, p unsafe.Pointer);

// The 'instructions' of the decoding machine
type decInstr struct {
	op	decOp;
	field		int;	// field number
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

func decBool(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(bool));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*bool)(p) = DecodeInt(state) != 0;
}

func decInt(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int)(p) = int(DecodeInt(state));
}

func decUint(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint)(p) = uint(DecodeUint(state));
}

func decInt8(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int8));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int8)(p) = int8(DecodeInt(state));
}

func decUint8(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint8));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint8)(p) = uint8(DecodeUint(state));
}

func decInt16(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int16));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int16)(p) = int16(DecodeInt(state));
}

func decUint16(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint16));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint16)(p) = uint16(DecodeUint(state));
}

func decInt32(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int32));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int32)(p) = int32(DecodeInt(state));
}

func decUint32(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint32));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint32)(p) = uint32(DecodeUint(state));
}

func decInt64(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int64));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*int64)(p) = int64(DecodeInt(state));
}

func decUint64(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint64));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*uint64)(p) = uint64(DecodeUint(state));
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

func decFloat(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*float)(p) = float(floatFromBits(uint64(DecodeUint(state))));
}

func decFloat32(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float32));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*float32)(p) = float32(floatFromBits(uint64(DecodeUint(state))));
}

func decFloat64(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float64));
		}
		p = *(*unsafe.Pointer)(p);
	}
	*(*float64)(p) = floatFromBits(uint64(DecodeUint(state)));
}

// uint8 arrays are encoded as an unsigned count followed by the raw bytes.
func decUint8Array(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new([]uint8));
		}
		p = *(*unsafe.Pointer)(p);
	}
	b := make([]uint8, DecodeUint(state));
	state.r.Read(b);
	*(*[]uint8)(p) = b;
}

// Strings are encoded as an unsigned count followed by the raw bytes.
func decString(i *decInstr, state *DecState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new([]byte));
		}
		p = *(*unsafe.Pointer)(p);
	}
	b := make([]byte, DecodeUint(state));
	state.r.Read(b);
	*(*string)(p) = string(b);
}

// Execution engine

// The encoder engine is an array of instructions indexed by field number of the incoming
// data.  It is executed with random access according to field number.
type decEngine struct {
	instr	[]decInstr
}

func decodeStruct(engine *decEngine, rtyp reflect.StructType, r io.Reader, p uintptr, indir int) os.Error {
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
	state := new(DecState);
	state.r = r;
	state.fieldnum = -1;
	basep := p;
	for state.err == nil {
		delta := int(DecodeUint(state));
		if delta < 0 {
			state.err = os.ErrorString("gob decode: corrupted data: negative delta");
			break
		}
		if state.err != nil || delta == 0 {	// struct terminator is zero delta fieldnum
			break
		}
		fieldnum := state.fieldnum + delta;
		if fieldnum >= len(engine.instr) {
			panicln("TODO(r): need to handle unknown data");
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

func decodeArrayHelper(state *DecState, p uintptr, elemOp decOp, elemWid, length, elemIndir int) os.Error {
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

func decodeArray(atyp reflect.ArrayType, state *DecState, p uintptr, elemOp decOp, elemWid, length, indir, elemIndir int) os.Error {
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
	if DecodeUint(state) != uint64(length) {
		return os.ErrorString("length mismatch in decodeArray");
	}
	return decodeArrayHelper(state, p, elemOp, elemWid, length, elemIndir);
}

func decodeSlice(atyp reflect.ArrayType, state *DecState, p uintptr, elemOp decOp, elemWid, indir, elemIndir int) os.Error {
	length := int(DecodeUint(state));
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
	return decodeArrayHelper(state, hdrp.Data, elemOp, elemWid, length, elemIndir);
}

var decEngineMap = make(map[reflect.Type] *decEngine)
var decOpMap = map[int] decOp {
	 reflect.BoolKind: decBool,
	 reflect.IntKind: decInt,
	 reflect.Int8Kind: decInt8,
	 reflect.Int16Kind: decInt16,
	 reflect.Int32Kind: decInt32,
	 reflect.Int64Kind: decInt64,
	 reflect.UintKind: decUint,
	 reflect.Uint8Kind: decUint8,
	 reflect.Uint16Kind: decUint16,
	 reflect.Uint32Kind: decUint32,
	 reflect.Uint64Kind: decUint64,
	 reflect.FloatKind: decFloat,
	 reflect.Float32Kind: decFloat32,
	 reflect.Float64Kind: decFloat64,
	 reflect.StringKind: decString,
}

func getDecEngine(rt reflect.Type) *decEngine

func decOpFor(typ reflect.Type) decOp {
	op, ok := decOpMap[typ.Kind()];
	if !ok {
		// Special cases
		if typ.Kind() == reflect.ArrayKind {
			atyp := typ.(reflect.ArrayType);
			switch {
			case atyp.Elem().Kind() == reflect.Uint8Kind:
				op = decUint8Array
			case atyp.IsSlice():
				elemOp := decOpFor(atyp.Elem());
				_, elemIndir := indirect(atyp.Elem());
				op = func(i *decInstr, state *DecState, p unsafe.Pointer) {
					state.err = decodeSlice(atyp, state, uintptr(p), elemOp, atyp.Elem().Size(), i.indir, elemIndir);
				};
			case !atyp.IsSlice():
				elemOp := decOpFor(atyp.Elem());
				_, elemIndir := indirect(atyp.Elem());
				op = func(i *decInstr, state *DecState, p unsafe.Pointer) {
					state.err = decodeArray(atyp, state, uintptr(p), elemOp, atyp.Elem().Size(), atyp.Len(), i.indir, elemIndir);
				};
			}
		}
		if typ.Kind() == reflect.StructKind {
			// Generate a closure that calls out to the engine for the nested type.
			engine := getDecEngine(typ);
			styp := typ.(reflect.StructType);
			op = func(i *decInstr, state *DecState, p unsafe.Pointer) {
				state.err = decodeStruct(engine, styp, state.r, uintptr(p), i.indir)
			};
		}
	}
	if op == nil {
		panicln("decode can't handle type", typ.String());
	}
	return op
}

func compileDec(rt reflect.Type, typ Type) *decEngine {
	srt, ok1 := rt.(reflect.StructType);
	styp, ok2 := typ.(*structType);
	if !ok1 || !ok2 {
		panicln("TODO: can't handle non-structs");
	}
	engine := new(decEngine);
	engine.instr = make([]decInstr, len(styp.field));
	for fieldnum := 0; fieldnum < len(styp.field); fieldnum++ {
		field := styp.field[fieldnum];
		// TODO(r): verify compatibility with corresponding field of data.
		// For now, assume perfect correspondence between struct and gob.
		_name, ftyp, _tag, offset := srt.Field(fieldnum);
		// How many indirections to the underlying data?
		indir := 0;
		for {
			pt, ok := ftyp.(reflect.PtrType);
			if !ok {
				break
			}
			ftyp = pt.Sub();
			indir++;
		}
		op := decOpFor(ftyp);
		engine.instr[fieldnum] = decInstr{op, fieldnum, indir, uintptr(offset)};
	}
	return engine;
}


func getDecEngine(rt reflect.Type) *decEngine {
	engine, ok := decEngineMap[rt];
	if !ok {
		return compileDec(rt, newType(rt.Name(), rt));
		decEngineMap[rt] = engine;
	}
	return engine;
}

func Decode(r io.Reader, e interface{}) os.Error {
	// Dereference down to the underlying object.
	rt, indir := indirect(reflect.Typeof(e));
	v := reflect.NewValue(e);
	for i := 0; i < indir; i++ {
		v = reflect.Indirect(v);
	}
	if rt.Kind() != reflect.StructKind {
		return os.ErrorString("decode can't handle " + rt.String())
	}
	typeLock.Lock();
	engine := getDecEngine(rt);
	typeLock.Unlock();
	return decodeStruct(engine, rt.(reflect.StructType), r, uintptr(v.Addr()), 0);
}
