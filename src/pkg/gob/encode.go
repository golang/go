// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"io"
	"math"
	"os"
	"reflect"
	"unsafe"
)

const uint64Size = unsafe.Sizeof(uint64(0))

// The global execution state of an instance of the encoder.
// Field numbers are delta encoded and always increase. The field
// number is initialized to -1 so 0 comes out as delta(1). A delta of
// 0 terminates the structure.
type encoderState struct {
	enc      *Encoder
	b        *bytes.Buffer
	sendZero bool                 // encoding an array element or map key/value pair; send zero values
	fieldnum int                  // the last field number written.
	buf      [1 + uint64Size]byte // buffer used by the encoder; here to avoid allocation.
}

func newEncoderState(enc *Encoder, b *bytes.Buffer) *encoderState {
	return &encoderState{enc: enc, b: b}
}

// Unsigned integers have a two-state encoding.  If the number is less
// than 128 (0 through 0x7F), its value is written directly.
// Otherwise the value is written in big-endian byte order preceded
// by the byte length, negated.

// encodeUint writes an encoded unsigned integer to state.b.
func (state *encoderState) encodeUint(x uint64) {
	if x <= 0x7F {
		err := state.b.WriteByte(uint8(x))
		if err != nil {
			error(err)
		}
		return
	}
	var n, m int
	m = uint64Size
	for n = 1; x > 0; n++ {
		state.buf[m] = uint8(x & 0xFF)
		x >>= 8
		m--
	}
	state.buf[m] = uint8(-(n - 1))
	n, err := state.b.Write(state.buf[m : uint64Size+1])
	if err != nil {
		error(err)
	}
}

// encodeInt writes an encoded signed integer to state.w.
// The low bit of the encoding says whether to bit complement the (other bits of the)
// uint to recover the int.
func (state *encoderState) encodeInt(i int64) {
	var x uint64
	if i < 0 {
		x = uint64(^i<<1) | 1
	} else {
		x = uint64(i << 1)
	}
	state.encodeUint(uint64(x))
}

type encOp func(i *encInstr, state *encoderState, p unsafe.Pointer)

// The 'instructions' of the encoding machine
type encInstr struct {
	op     encOp
	field  int     // field number
	indir  int     // how many pointer indirections to reach the value in the struct
	offset uintptr // offset in the structure of the field to encode
}

// Emit a field number and update the state to record its value for delta encoding.
// If the instruction pointer is nil, do nothing
func (state *encoderState) update(instr *encInstr) {
	if instr != nil {
		state.encodeUint(uint64(instr.field - state.fieldnum))
		state.fieldnum = instr.field
	}
}

// Each encoder is responsible for handling any indirections associated
// with the data structure.  If any pointer so reached is nil, no bytes are written.
// If the data item is zero, no bytes are written.
// Otherwise, the output (for a scalar) is the field number, as an encoded integer,
// followed by the field data in its appropriate format.

func encIndirect(p unsafe.Pointer, indir int) unsafe.Pointer {
	for ; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p)
		if p == nil {
			return unsafe.Pointer(nil)
		}
	}
	return p
}

func encBool(i *encInstr, state *encoderState, p unsafe.Pointer) {
	b := *(*bool)(p)
	if b || state.sendZero {
		state.update(i)
		if b {
			state.encodeUint(1)
		} else {
			state.encodeUint(0)
		}
	}
}

func encInt(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeInt(v)
	}
}

func encUint(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeUint(v)
	}
}

func encInt8(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int8)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeInt(v)
	}
}

func encUint8(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint8)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeUint(v)
	}
}

func encInt16(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int16)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeInt(v)
	}
}

func encUint16(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint16)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeUint(v)
	}
}

func encInt32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int32)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeInt(v)
	}
}

func encUint32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint32)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeUint(v)
	}
}

func encInt64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := *(*int64)(p)
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeInt(v)
	}
}

func encUint64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := *(*uint64)(p)
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeUint(v)
	}
}

func encUintptr(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uintptr)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		state.encodeUint(v)
	}
}

// Floating-point numbers are transmitted as uint64s holding the bits
// of the underlying representation.  They are sent byte-reversed, with
// the exponent end coming out first, so integer floating point numbers
// (for example) transmit more compactly.  This routine does the
// swizzling.
func floatBits(f float64) uint64 {
	u := math.Float64bits(f)
	var v uint64
	for i := 0; i < 8; i++ {
		v <<= 8
		v |= u & 0xFF
		u >>= 8
	}
	return v
}

func encFloat32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	f := *(*float32)(p)
	if f != 0 || state.sendZero {
		v := floatBits(float64(f))
		state.update(i)
		state.encodeUint(v)
	}
}

func encFloat64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	f := *(*float64)(p)
	if f != 0 || state.sendZero {
		state.update(i)
		v := floatBits(f)
		state.encodeUint(v)
	}
}

// Complex numbers are just a pair of floating-point numbers, real part first.
func encComplex64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	c := *(*complex64)(p)
	if c != 0+0i || state.sendZero {
		rpart := floatBits(float64(real(c)))
		ipart := floatBits(float64(imag(c)))
		state.update(i)
		state.encodeUint(rpart)
		state.encodeUint(ipart)
	}
}

func encComplex128(i *encInstr, state *encoderState, p unsafe.Pointer) {
	c := *(*complex128)(p)
	if c != 0+0i || state.sendZero {
		rpart := floatBits(real(c))
		ipart := floatBits(imag(c))
		state.update(i)
		state.encodeUint(rpart)
		state.encodeUint(ipart)
	}
}

func encNoOp(i *encInstr, state *encoderState, p unsafe.Pointer) {
}

// Byte arrays are encoded as an unsigned count followed by the raw bytes.
func encUint8Array(i *encInstr, state *encoderState, p unsafe.Pointer) {
	b := *(*[]byte)(p)
	if len(b) > 0 || state.sendZero {
		state.update(i)
		state.encodeUint(uint64(len(b)))
		state.b.Write(b)
	}
}

// Strings are encoded as an unsigned count followed by the raw bytes.
func encString(i *encInstr, state *encoderState, p unsafe.Pointer) {
	s := *(*string)(p)
	if len(s) > 0 || state.sendZero {
		state.update(i)
		state.encodeUint(uint64(len(s)))
		io.WriteString(state.b, s)
	}
}

// The end of a struct is marked by a delta field number of 0.
func encStructTerminator(i *encInstr, state *encoderState, p unsafe.Pointer) {
	state.encodeUint(0)
}

// Execution engine

// The encoder engine is an array of instructions indexed by field number of the encoding
// data, typically a struct.  It is executed top to bottom, walking the struct.
type encEngine struct {
	instr []encInstr
}

const singletonField = 0

func (enc *Encoder) encodeSingle(b *bytes.Buffer, engine *encEngine, basep uintptr) {
	state := newEncoderState(enc, b)
	state.fieldnum = singletonField
	// There is no surrounding struct to frame the transmission, so we must
	// generate data even if the item is zero.  To do this, set sendZero.
	state.sendZero = true
	instr := &engine.instr[singletonField]
	p := unsafe.Pointer(basep) // offset will be zero
	if instr.indir > 0 {
		if p = encIndirect(p, instr.indir); p == nil {
			return
		}
	}
	instr.op(instr, state, p)
}

func (enc *Encoder) encodeStruct(b *bytes.Buffer, engine *encEngine, basep uintptr) {
	state := newEncoderState(enc, b)
	state.fieldnum = -1
	for i := 0; i < len(engine.instr); i++ {
		instr := &engine.instr[i]
		p := unsafe.Pointer(basep + instr.offset)
		if instr.indir > 0 {
			if p = encIndirect(p, instr.indir); p == nil {
				continue
			}
		}
		instr.op(instr, state, p)
	}
}

func (enc *Encoder) encodeArray(b *bytes.Buffer, p uintptr, op encOp, elemWid uintptr, elemIndir int, length int) {
	state := newEncoderState(enc, b)
	state.fieldnum = -1
	state.sendZero = true
	state.encodeUint(uint64(length))
	for i := 0; i < length; i++ {
		elemp := p
		up := unsafe.Pointer(elemp)
		if elemIndir > 0 {
			if up = encIndirect(up, elemIndir); up == nil {
				errorf("gob: encodeArray: nil element")
			}
			elemp = uintptr(up)
		}
		op(nil, state, unsafe.Pointer(elemp))
		p += uintptr(elemWid)
	}
}

func encodeReflectValue(state *encoderState, v reflect.Value, op encOp, indir int) {
	for i := 0; i < indir && v != nil; i++ {
		v = reflect.Indirect(v)
	}
	if v == nil {
		errorf("gob: encodeReflectValue: nil element")
	}
	op(nil, state, unsafe.Pointer(v.Addr()))
}

func (enc *Encoder) encodeMap(b *bytes.Buffer, mv *reflect.MapValue, keyOp, elemOp encOp, keyIndir, elemIndir int) {
	state := newEncoderState(enc, b)
	state.fieldnum = -1
	state.sendZero = true
	keys := mv.Keys()
	state.encodeUint(uint64(len(keys)))
	for _, key := range keys {
		encodeReflectValue(state, key, keyOp, keyIndir)
		encodeReflectValue(state, mv.Elem(key), elemOp, elemIndir)
	}
}

// To send an interface, we send a string identifying the concrete type, followed
// by the type identifier (which might require defining that type right now), followed
// by the concrete value.  A nil value gets sent as the empty string for the name,
// followed by no value.
func (enc *Encoder) encodeInterface(b *bytes.Buffer, iv *reflect.InterfaceValue) {
	state := newEncoderState(enc, b)
	state.fieldnum = -1
	state.sendZero = true
	if iv.IsNil() {
		state.encodeUint(0)
		return
	}

	typ, _ := indirect(iv.Elem().Type())
	name, ok := concreteTypeToName[typ]
	if !ok {
		errorf("gob: type not registered for interface: %s", typ)
	}
	// Send the name.
	state.encodeUint(uint64(len(name)))
	_, err := io.WriteString(state.b, name)
	if err != nil {
		error(err)
	}
	// Send (and maybe first define) the type id.
	enc.sendTypeDescriptor(typ)
	// Encode the value into a new buffer.
	data := new(bytes.Buffer)
	err = enc.encode(data, iv.Elem())
	if err != nil {
		error(err)
	}
	state.encodeUint(uint64(data.Len()))
	_, err = state.b.Write(data.Bytes())
	if err != nil {
		error(err)
	}
}

var encOpMap = []encOp{
	reflect.Bool:       encBool,
	reflect.Int:        encInt,
	reflect.Int8:       encInt8,
	reflect.Int16:      encInt16,
	reflect.Int32:      encInt32,
	reflect.Int64:      encInt64,
	reflect.Uint:       encUint,
	reflect.Uint8:      encUint8,
	reflect.Uint16:     encUint16,
	reflect.Uint32:     encUint32,
	reflect.Uint64:     encUint64,
	reflect.Uintptr:    encUintptr,
	reflect.Float32:    encFloat32,
	reflect.Float64:    encFloat64,
	reflect.Complex64:  encComplex64,
	reflect.Complex128: encComplex128,
	reflect.String:     encString,
}

// Return the encoding op for the base type under rt and
// the indirection count to reach it.
func (enc *Encoder) encOpFor(rt reflect.Type) (encOp, int) {
	typ, indir := indirect(rt)
	var op encOp
	k := typ.Kind()
	if int(k) < len(encOpMap) {
		op = encOpMap[k]
	}
	if op == nil {
		// Special cases
		switch t := typ.(type) {
		case *reflect.SliceType:
			if t.Elem().Kind() == reflect.Uint8 {
				op = encUint8Array
				break
			}
			// Slices have a header; we decode it to find the underlying array.
			elemOp, indir := enc.encOpFor(t.Elem())
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				slice := (*reflect.SliceHeader)(p)
				if !state.sendZero && slice.Len == 0 {
					return
				}
				state.update(i)
				state.enc.encodeArray(state.b, slice.Data, elemOp, t.Elem().Size(), indir, int(slice.Len))
			}
		case *reflect.ArrayType:
			// True arrays have size in the type.
			elemOp, indir := enc.encOpFor(t.Elem())
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				state.update(i)
				state.enc.encodeArray(state.b, uintptr(p), elemOp, t.Elem().Size(), indir, t.Len())
			}
		case *reflect.MapType:
			keyOp, keyIndir := enc.encOpFor(t.Key())
			elemOp, elemIndir := enc.encOpFor(t.Elem())
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				// Maps cannot be accessed by moving addresses around the way
				// that slices etc. can.  We must recover a full reflection value for
				// the iteration.
				v := reflect.NewValue(unsafe.Unreflect(t, unsafe.Pointer((p))))
				mv := reflect.Indirect(v).(*reflect.MapValue)
				if !state.sendZero && mv.Len() == 0 {
					return
				}
				state.update(i)
				state.enc.encodeMap(state.b, mv, keyOp, elemOp, keyIndir, elemIndir)
			}
		case *reflect.StructType:
			// Generate a closure that calls out to the engine for the nested type.
			enc.getEncEngine(typ)
			info := mustGetTypeInfo(typ)
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				state.update(i)
				// indirect through info to delay evaluation for recursive structs
				state.enc.encodeStruct(state.b, info.encoder, uintptr(p))
			}
		case *reflect.InterfaceType:
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				// Interfaces transmit the name and contents of the concrete
				// value they contain.
				v := reflect.NewValue(unsafe.Unreflect(t, unsafe.Pointer((p))))
				iv := reflect.Indirect(v).(*reflect.InterfaceValue)
				if !state.sendZero && (iv == nil || iv.IsNil()) {
					return
				}
				state.update(i)
				state.enc.encodeInterface(state.b, iv)
			}
		}
	}
	if op == nil {
		errorf("gob enc: can't happen: encode type %s", rt.String())
	}
	return op, indir
}

// The local Type was compiled from the actual value, so we know it's compatible.
func (enc *Encoder) compileEnc(rt reflect.Type) *encEngine {
	srt, isStruct := rt.(*reflect.StructType)
	engine := new(encEngine)
	if isStruct {
		engine.instr = make([]encInstr, srt.NumField()+1) // +1 for terminator
		for fieldnum := 0; fieldnum < srt.NumField(); fieldnum++ {
			f := srt.Field(fieldnum)
			op, indir := enc.encOpFor(f.Type)
			if !isExported(f.Name) {
				op = encNoOp
			}
			engine.instr[fieldnum] = encInstr{op, fieldnum, indir, uintptr(f.Offset)}
		}
		engine.instr[srt.NumField()] = encInstr{encStructTerminator, 0, 0, 0}
	} else {
		engine.instr = make([]encInstr, 1)
		op, indir := enc.encOpFor(rt)
		engine.instr[0] = encInstr{op, singletonField, indir, 0} // offset is zero
	}
	return engine
}

// typeLock must be held (or we're in initialization and guaranteed single-threaded).
// The reflection type must have all its indirections processed out.
func (enc *Encoder) getEncEngine(rt reflect.Type) *encEngine {
	info, err1 := getTypeInfo(rt)
	if err1 != nil {
		error(err1)
	}
	if info.encoder == nil {
		// mark this engine as underway before compiling to handle recursive types.
		info.encoder = new(encEngine)
		info.encoder = enc.compileEnc(rt)
	}
	return info.encoder
}

// Put this in a function so we can hold the lock only while compiling, not when encoding.
func (enc *Encoder) lockAndGetEncEngine(rt reflect.Type) *encEngine {
	typeLock.Lock()
	defer typeLock.Unlock()
	return enc.getEncEngine(rt)
}

func (enc *Encoder) encode(b *bytes.Buffer, value reflect.Value) (err os.Error) {
	defer catchError(&err)
	// Dereference down to the underlying object.
	rt, indir := indirect(value.Type())
	for i := 0; i < indir; i++ {
		value = reflect.Indirect(value)
	}
	engine := enc.lockAndGetEncEngine(rt)
	if value.Type().Kind() == reflect.Struct {
		enc.encodeStruct(b, engine, value.Addr())
	} else {
		enc.encodeSingle(b, engine, value.Addr())
	}
	return nil
}
