// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes";
	"gob";
	"io";
	"math";
	"os";
	"reflect";
	"sync";
	"unsafe";
)

// The global execution state of an instance of the encoder.
// Field numbers are delta encoded and always increase. The field
// number is initialized to -1 so 0 comes out as delta(1). A delta of
// 0 terminates the structure.
type encoderState struct {
	b	*bytes.Buffer;
	err	os.Error;	// error encountered during encoding;
	fieldnum	int;	// the last field number written.
	buf [16]byte;	// buffer used by the encoder; here to avoid allocation.
}

// Integers encode as a variant of Google's protocol buffer varint (varvarint?).
// The variant is that the continuation bytes have a zero top bit instead of a one.
// That way there's only one bit to clear and the value is a little easier to see if
// you're the unfortunate sort of person who must read the hex to debug.

// encodeUint writes an encoded unsigned integer to state.b.  Sets state.err.
// If state.err is already non-nil, it does nothing.
func encodeUint(state *encoderState, x uint64) {
	var n int;
	if state.err != nil {
		return
	}
	for n = 0; x > 0x7F; n++ {
		state.buf[n] = uint8(x & 0x7F);
		x >>= 7;
	}
	state.buf[n] = 0x80 | uint8(x);
	n, state.err = state.b.Write(state.buf[0:n+1]);
}

// encodeInt writes an encoded signed integer to state.w.
// The low bit of the encoding says whether to bit complement the (other bits of the) uint to recover the int.
// Sets state.err. If state.err is already non-nil, it does nothing.
func encodeInt(state *encoderState, i int64){
	var x uint64;
	if i < 0 {
		x = uint64(^i << 1) | 1
	} else {
		x = uint64(i << 1)
	}
	encodeUint(state, uint64(x))
}

type encInstr struct
type encOp func(i *encInstr, state *encoderState, p unsafe.Pointer)

// The 'instructions' of the encoding machine
type encInstr struct {
	op	encOp;
	field		int;	// field number
	indir	int;	// how many pointer indirections to reach the value in the struct
	offset	uintptr;	// offset in the structure of the field to encode
}

// Emit a field number and update the state to record its value for delta encoding.
// If the instruction pointer is nil, do nothing
func (state *encoderState) update(instr *encInstr) {
	if instr != nil {
		encodeUint(state, uint64(instr.field - state.fieldnum));
		state.fieldnum = instr.field;
	}
}

// Each encoder is responsible for handling any indirections associated
// with the data structure.  If any pointer so reached is nil, no bytes are written.
// If the data item is zero, no bytes are written.
// Otherwise, the output (for a scalar) is the field number, as an encoded integer,
// followed by the field data in its appropriate format.

func encIndirect(p unsafe.Pointer, indir int) unsafe.Pointer {
	for ; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return unsafe.Pointer(nil)
		}
	}
	return p
}

func encBool(i *encInstr, state *encoderState, p unsafe.Pointer) {
	b := *(*bool)(p);
	if b {
		state.update(i);
		encodeUint(state, 1);
	}
}

func encInt(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int)(p));
	if v != 0 {
		state.update(i);
		encodeInt(state, v);
	}
}

func encUint(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint)(p));
	if v != 0 {
		state.update(i);
		encodeUint(state, v);
	}
}

func encInt8(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int8)(p));
	if v != 0 {
		state.update(i);
		encodeInt(state, v);
	}
}

func encUint8(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint8)(p));
	if v != 0 {
		state.update(i);
		encodeUint(state, v);
	}
}

func encInt16(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int16)(p));
	if v != 0 {
		state.update(i);
		encodeInt(state, v);
	}
}

func encUint16(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint16)(p));
	if v != 0 {
		state.update(i);
		encodeUint(state, v);
	}
}

func encInt32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int32)(p));
	if v != 0 {
		state.update(i);
		encodeInt(state, v);
	}
}

func encUint32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint32)(p));
	if v != 0 {
		state.update(i);
		encodeUint(state, v);
	}
}

func encInt64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := *(*int64)(p);
	if v != 0 {
		state.update(i);
		encodeInt(state, v);
	}
}

func encUint64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := *(*uint64)(p);
	if v != 0 {
		state.update(i);
		encodeUint(state, v);
	}
}

func encUintptr(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uintptr)(p));
	if v != 0 {
		state.update(i);
		encodeUint(state, v);
	}
}

// Floating-point numbers are transmitted as uint64s holding the bits
// of the underlying representation.  They are sent byte-reversed, with
// the exponent end coming out first, so integer floating point numbers
// (for example) transmit more compactly.  This routine does the
// swizzling.
func floatBits(f float64) uint64 {
	u := math.Float64bits(f);
	var v uint64;
	for i := 0; i < 8; i++ {
		v <<= 8;
		v |= u & 0xFF;
		u >>= 8;
	}
	return v;
}

func encFloat(i *encInstr, state *encoderState, p unsafe.Pointer) {
	f := float(*(*float)(p));
	if f != 0 {
		v := floatBits(float64(f));
		state.update(i);
		encodeUint(state, v);
	}
}

func encFloat32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	f := float32(*(*float32)(p));
	if f != 0 {
		v := floatBits(float64(f));
		state.update(i);
		encodeUint(state, v);
	}
}

func encFloat64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	f := *(*float64)(p);
	if f != 0 {
		state.update(i);
		v := floatBits(f);
		encodeUint(state, v);
	}
}

// Byte arrays are encoded as an unsigned count followed by the raw bytes.
func encUint8Array(i *encInstr, state *encoderState, p unsafe.Pointer) {
	b := *(*[]byte)(p);
	if len(b) > 0 {
		state.update(i);
		encodeUint(state, uint64(len(b)));
		state.b.Write(b);
	}
}

// Strings are encoded as an unsigned count followed by the raw bytes.
func encString(i *encInstr, state *encoderState, p unsafe.Pointer) {
	s := *(*string)(p);
	if len(s) > 0 {
		state.update(i);
		encodeUint(state, uint64(len(s)));
		io.WriteString(state.b, s);
	}
}

// The end of a struct is marked by a delta field number of 0.
func encStructTerminator(i *encInstr, state *encoderState, p unsafe.Pointer) {
	encodeUint(state, 0);
}

// Execution engine

// The encoder engine is an array of instructions indexed by field number of the encoding
// data, typically a struct.  It is executed top to bottom, walking the struct.
type encEngine struct {
	instr	[]encInstr
}

func encodeStruct(engine *encEngine, b *bytes.Buffer, basep uintptr) os.Error {
	state := new(encoderState);
	state.b = b;
	state.fieldnum = -1;
	for i := 0; i < len(engine.instr); i++ {
		instr := &engine.instr[i];
		p := unsafe.Pointer(basep+instr.offset);
		if instr.indir > 0 {
			if p = encIndirect(p, instr.indir); p == nil {
				continue
			}
		}
		instr.op(instr, state, p);
		if state.err != nil {
			break
		}
	}
	return state.err
}

func encodeArray(b *bytes.Buffer, p uintptr, op encOp, elemWid uintptr, length int, elemIndir int) os.Error {
	state := new(encoderState);
	state.b = b;
	state.fieldnum = -1;
	encodeUint(state, uint64(length));
	for i := 0; i < length && state.err == nil; i++ {
		elemp := p;
		up := unsafe.Pointer(elemp);
		if elemIndir > 0 {
			if up = encIndirect(up, elemIndir); up == nil {
				state.err = os.ErrorString("gob: encodeArray: nil element");
				break
			}
			elemp = uintptr(up);
		}
		op(nil, state, unsafe.Pointer(elemp));
		p += uintptr(elemWid);
	}
	return state.err
}

var encOpMap = map[reflect.Type] encOp {
	reflect.Typeof((*reflect.BoolType)(nil)): encBool,
	reflect.Typeof((*reflect.IntType)(nil)): encInt,
	reflect.Typeof((*reflect.Int8Type)(nil)): encInt8,
	reflect.Typeof((*reflect.Int16Type)(nil)): encInt16,
	reflect.Typeof((*reflect.Int32Type)(nil)): encInt32,
	reflect.Typeof((*reflect.Int64Type)(nil)): encInt64,
	reflect.Typeof((*reflect.UintType)(nil)): encUint,
	reflect.Typeof((*reflect.Uint8Type)(nil)): encUint8,
	reflect.Typeof((*reflect.Uint16Type)(nil)): encUint16,
	reflect.Typeof((*reflect.Uint32Type)(nil)): encUint32,
	reflect.Typeof((*reflect.Uint64Type)(nil)): encUint64,
	reflect.Typeof((*reflect.UintptrType)(nil)): encUintptr,
	reflect.Typeof((*reflect.FloatType)(nil)): encFloat,
	reflect.Typeof((*reflect.Float32Type)(nil)): encFloat32,
	reflect.Typeof((*reflect.Float64Type)(nil)): encFloat64,
	reflect.Typeof((*reflect.StringType)(nil)): encString,
}

func getEncEngine(rt reflect.Type) *encEngine

// Return the encoding op for the base type under rt and
// the indirection count to reach it.
func encOpFor(rt reflect.Type) (encOp, int) {
	typ, indir := indirect(rt);
	op, ok := encOpMap[reflect.Typeof(typ)];
	if !ok {
		typ, _ := indirect(rt);
		// Special cases
		switch t := typ.(type) {
		case *reflect.SliceType:
			if _, ok := t.Elem().(*reflect.Uint8Type); ok {
				op = encUint8Array;
				break;
			}
			// Slices have a header; we decode it to find the underlying array.
			elemOp, indir := encOpFor(t.Elem());
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				slice := (*reflect.SliceHeader)(p);
				if slice.Len == 0 {
					return
				}
				state.update(i);
				state.err = encodeArray(state.b, slice.Data, elemOp, t.Elem().Size(), int(slice.Len), indir);
			};
		case *reflect.ArrayType:
			// True arrays have size in the type.
			elemOp, indir := encOpFor(t.Elem());
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				state.update(i);
				state.err = encodeArray(state.b, uintptr(p), elemOp, t.Elem().Size(), t.Len(), indir);
			};
		case *reflect.StructType:
			// Generate a closure that calls out to the engine for the nested type.
			engine := getEncEngine(typ);
			info := getTypeInfo(typ);
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				state.update(i);
				// indirect through info to delay evaluation for recursive structs
				state.err = encodeStruct(info.encoder, state.b, uintptr(p));
			};
		}
	}
	if op == nil {
		panicln("encode can't handle type", rt.String());
	}
	return op, indir
}

// The local Type was compiled from the actual value, so we know it's compatible.
func compileEnc(rt reflect.Type) *encEngine {
	srt, ok := rt.(*reflect.StructType);
	if !ok {
		panicln("TODO: can't handle non-structs");
	}
	engine := new(encEngine);
	engine.instr = make([]encInstr, srt.NumField()+1);	// +1 for terminator
	for fieldnum := 0; fieldnum < srt.NumField(); fieldnum++ {
		f := srt.Field(fieldnum);
		op, indir := encOpFor(f.Type);
		engine.instr[fieldnum] = encInstr{op, fieldnum, indir, uintptr(f.Offset)};
	}
	engine.instr[srt.NumField()] = encInstr{encStructTerminator, 0, 0, 0};
	return engine;
}

// typeLock must be held (or we're in initialization and guaranteed single-threaded).
// The reflection type must have all its indirections processed out.
func getEncEngine(rt reflect.Type) *encEngine {
	info := getTypeInfo(rt);
	if info.encoder == nil {
		// mark this engine as underway before compiling to handle recursive types.
		info.encoder = new(encEngine);
		info.encoder = compileEnc(rt);
	}
	return info.encoder;
}

func encode(b *bytes.Buffer, e interface{}) os.Error {
	// Dereference down to the underlying object.
	rt, indir := indirect(reflect.Typeof(e));
	v := reflect.NewValue(e);
	for i := 0; i < indir; i++ {
		v = reflect.Indirect(v);
	}
	if _, ok := v.(*reflect.StructValue); !ok {
		return os.ErrorString("gob: encode can't handle " + v.Type().String())
	}
	typeLock.Lock();
	engine := getEncEngine(rt);
	typeLock.Unlock();
	return encodeStruct(engine, b, v.Addr());
}
