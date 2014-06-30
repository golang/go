// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"encoding"
	"math"
	"reflect"
)

const uint64Size = 8

// encoderState is the global execution state of an instance of the encoder.
// Field numbers are delta encoded and always increase. The field
// number is initialized to -1 so 0 comes out as delta(1). A delta of
// 0 terminates the structure.
type encoderState struct {
	enc      *Encoder
	b        *bytes.Buffer
	sendZero bool                 // encoding an array element or map key/value pair; send zero values
	fieldnum int                  // the last field number written.
	buf      [1 + uint64Size]byte // buffer used by the encoder; here to avoid allocation.
	next     *encoderState        // for free list
}

func (enc *Encoder) newEncoderState(b *bytes.Buffer) *encoderState {
	e := enc.freeList
	if e == nil {
		e = new(encoderState)
		e.enc = enc
	} else {
		enc.freeList = e.next
	}
	e.sendZero = false
	e.fieldnum = 0
	e.b = b
	return e
}

func (enc *Encoder) freeEncoderState(e *encoderState) {
	e.next = enc.freeList
	enc.freeList = e
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
			error_(err)
		}
		return
	}
	i := uint64Size
	for x > 0 {
		state.buf[i] = uint8(x)
		x >>= 8
		i--
	}
	state.buf[i] = uint8(i - uint64Size) // = loop count, negated
	_, err := state.b.Write(state.buf[i : uint64Size+1])
	if err != nil {
		error_(err)
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

// encOp is the signature of an encoding operator for a given type.
type encOp func(i *encInstr, state *encoderState, v reflect.Value)

// The 'instructions' of the encoding machine
type encInstr struct {
	op    encOp
	field int   // field number in input
	index []int // struct index
	indir int   // how many pointer indirections to reach the value in the struct
}

// update emits a field number and updates the state to record its value for delta encoding.
// If the instruction pointer is nil, it does nothing
func (state *encoderState) update(instr *encInstr) {
	if instr != nil {
		state.encodeUint(uint64(instr.field - state.fieldnum))
		state.fieldnum = instr.field
	}
}

// Each encoder for a composite is responsible for handling any
// indirections associated with the elements of the data structure.
// If any pointer so reached is nil, no bytes are written.  If the
// data item is zero, no bytes are written.  Single values - ints,
// strings etc. - are indirected before calling their encoders.
// Otherwise, the output (for a scalar) is the field number, as an
// encoded integer, followed by the field data in its appropriate
// format.

// encIndirect dereferences pv indir times and returns the result.
func encIndirect(pv reflect.Value, indir int) reflect.Value {
	for ; indir > 0; indir-- {
		if pv.IsNil() {
			break
		}
		pv = pv.Elem()
	}
	return pv
}

// encBool encodes the bool referenced by v as an unsigned 0 or 1.
func encBool(i *encInstr, state *encoderState, v reflect.Value) {
	b := v.Bool()
	if b || state.sendZero {
		state.update(i)
		if b {
			state.encodeUint(1)
		} else {
			state.encodeUint(0)
		}
	}
}

// encInt encodes the signed integer (int int8 int16 int32 int64) referenced by v.
func encInt(i *encInstr, state *encoderState, v reflect.Value) {
	value := v.Int()
	if value != 0 || state.sendZero {
		state.update(i)
		state.encodeInt(value)
	}
}

// encUint encodes the unsigned integer (uint uint8 uint16 uint32 uint64 uintptr) referenced by v.
func encUint(i *encInstr, state *encoderState, v reflect.Value) {
	value := v.Uint()
	if value != 0 || state.sendZero {
		state.update(i)
		state.encodeUint(value)
	}
}

// floatBits returns a uint64 holding the bits of a floating-point number.
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

// encFloat encodes the floating point value (float32 float64) referenced by v.
func encFloat(i *encInstr, state *encoderState, v reflect.Value) {
	f := v.Float()
	if f != 0 || state.sendZero {
		bits := floatBits(f)
		state.update(i)
		state.encodeUint(bits)
	}
}

// encComplex encodes the complex value (complex64 complex128) referenced by v.
// Complex numbers are just a pair of floating-point numbers, real part first.
func encComplex(i *encInstr, state *encoderState, v reflect.Value) {
	c := v.Complex()
	if c != 0+0i || state.sendZero {
		rpart := floatBits(real(c))
		ipart := floatBits(imag(c))
		state.update(i)
		state.encodeUint(rpart)
		state.encodeUint(ipart)
	}
}

// encUint8Array encodes the byte array referenced by v.
// Byte arrays are encoded as an unsigned count followed by the raw bytes.
func encUint8Array(i *encInstr, state *encoderState, v reflect.Value) {
	b := v.Bytes()
	if len(b) > 0 || state.sendZero {
		state.update(i)
		state.encodeUint(uint64(len(b)))
		state.b.Write(b)
	}
}

// encString encodes the string referenced by v.
// Strings are encoded as an unsigned count followed by the raw bytes.
func encString(i *encInstr, state *encoderState, v reflect.Value) {
	s := v.String()
	if len(s) > 0 || state.sendZero {
		state.update(i)
		state.encodeUint(uint64(len(s)))
		state.b.WriteString(s)
	}
}

// encStructTerminator encodes the end of an encoded struct
// as delta field number of 0.
func encStructTerminator(i *encInstr, state *encoderState, v reflect.Value) {
	state.encodeUint(0)
}

// Execution engine

// encEngine an array of instructions indexed by field number of the encoding
// data, typically a struct.  It is executed top to bottom, walking the struct.
type encEngine struct {
	instr []encInstr
}

const singletonField = 0

// valid reports whether the value is valid and a non-nil pointer.
// (Slices, maps, and chans take care of themselves.)
func valid(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Invalid:
		return false
	case reflect.Ptr:
		return !v.IsNil()
	}
	return true
}

// encodeSingle encodes a single top-level non-struct value.
func (enc *Encoder) encodeSingle(b *bytes.Buffer, engine *encEngine, value reflect.Value) {
	state := enc.newEncoderState(b)
	defer enc.freeEncoderState(state)
	state.fieldnum = singletonField
	// There is no surrounding struct to frame the transmission, so we must
	// generate data even if the item is zero.  To do this, set sendZero.
	state.sendZero = true
	instr := &engine.instr[singletonField]
	if instr.indir > 0 {
		value = encIndirect(value, instr.indir)
	}
	if valid(value) {
		instr.op(instr, state, value)
	}
}

// encodeStruct encodes a single struct value.
func (enc *Encoder) encodeStruct(b *bytes.Buffer, engine *encEngine, value reflect.Value) {
	if !valid(value) {
		return
	}
	state := enc.newEncoderState(b)
	defer enc.freeEncoderState(state)
	state.fieldnum = -1
	for i := 0; i < len(engine.instr); i++ {
		instr := &engine.instr[i]
		if i >= value.NumField() {
			// encStructTerminator
			instr.op(instr, state, reflect.Value{})
			break
		}
		field := value.FieldByIndex(instr.index)
		if instr.indir > 0 {
			field = encIndirect(field, instr.indir)
		}
		if !valid(field) {
			continue
		}
		instr.op(instr, state, field)
	}
}

// encodeArray encodes the array whose 0th element is at p.
func (enc *Encoder) encodeArray(b *bytes.Buffer, value reflect.Value, op encOp, elemIndir int, length int) {
	state := enc.newEncoderState(b)
	defer enc.freeEncoderState(state)
	state.fieldnum = -1
	state.sendZero = true
	state.encodeUint(uint64(length))
	for i := 0; i < length; i++ {
		elem := value.Index(i)
		if elemIndir > 0 {
			elem = encIndirect(elem, elemIndir)
			if !valid(elem) {
				errorf("encodeArray: nil element")
			}
		}
		op(nil, state, elem)
	}
}

// encodeReflectValue is a helper for maps. It encodes the value v.
func encodeReflectValue(state *encoderState, v reflect.Value, op encOp, indir int) {
	for i := 0; i < indir && v.IsValid(); i++ {
		v = reflect.Indirect(v)
	}
	if !v.IsValid() {
		errorf("encodeReflectValue: nil element")
	}
	op(nil, state, v)
}

// encodeMap encodes a map as unsigned count followed by key:value pairs.
func (enc *Encoder) encodeMap(b *bytes.Buffer, mv reflect.Value, keyOp, elemOp encOp, keyIndir, elemIndir int) {
	state := enc.newEncoderState(b)
	state.fieldnum = -1
	state.sendZero = true
	keys := mv.MapKeys()
	state.encodeUint(uint64(len(keys)))
	for _, key := range keys {
		encodeReflectValue(state, key, keyOp, keyIndir)
		encodeReflectValue(state, mv.MapIndex(key), elemOp, elemIndir)
	}
	enc.freeEncoderState(state)
}

// encodeInterface encodes the interface value iv.
// To send an interface, we send a string identifying the concrete type, followed
// by the type identifier (which might require defining that type right now), followed
// by the concrete value.  A nil value gets sent as the empty string for the name,
// followed by no value.
func (enc *Encoder) encodeInterface(b *bytes.Buffer, iv reflect.Value) {
	// Gobs can encode nil interface values but not typed interface
	// values holding nil pointers, since nil pointers point to no value.
	elem := iv.Elem()
	if elem.Kind() == reflect.Ptr && elem.IsNil() {
		errorf("gob: cannot encode nil pointer of type %s inside interface", iv.Elem().Type())
	}
	state := enc.newEncoderState(b)
	state.fieldnum = -1
	state.sendZero = true
	if iv.IsNil() {
		state.encodeUint(0)
		return
	}

	ut := userType(iv.Elem().Type())
	registerLock.RLock()
	name, ok := concreteTypeToName[ut.base]
	registerLock.RUnlock()
	if !ok {
		errorf("type not registered for interface: %s", ut.base)
	}
	// Send the name.
	state.encodeUint(uint64(len(name)))
	_, err := state.b.WriteString(name)
	if err != nil {
		error_(err)
	}
	// Define the type id if necessary.
	enc.sendTypeDescriptor(enc.writer(), state, ut)
	// Send the type id.
	enc.sendTypeId(state, ut)
	// Encode the value into a new buffer.  Any nested type definitions
	// should be written to b, before the encoded value.
	enc.pushWriter(b)
	data := new(bytes.Buffer)
	data.Write(spaceForLength)
	enc.encode(data, elem, ut)
	if enc.err != nil {
		error_(enc.err)
	}
	enc.popWriter()
	enc.writeMessage(b, data)
	if enc.err != nil {
		error_(err)
	}
	enc.freeEncoderState(state)
}

// isZero reports whether the value is the zero of its type.
func isZero(val reflect.Value) bool {
	switch val.Kind() {
	case reflect.Array:
		for i := 0; i < val.Len(); i++ {
			if !isZero(val.Index(i)) {
				return false
			}
		}
		return true
	case reflect.Map, reflect.Slice, reflect.String:
		return val.Len() == 0
	case reflect.Bool:
		return !val.Bool()
	case reflect.Complex64, reflect.Complex128:
		return val.Complex() == 0
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Ptr:
		return val.IsNil()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return val.Int() == 0
	case reflect.Float32, reflect.Float64:
		return val.Float() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return val.Uint() == 0
	case reflect.Struct:
		for i := 0; i < val.NumField(); i++ {
			if !isZero(val.Field(i)) {
				return false
			}
		}
		return true
	}
	panic("unknown type in isZero " + val.Type().String())
}

// encGobEncoder encodes a value that implements the GobEncoder interface.
// The data is sent as a byte array.
func (enc *Encoder) encodeGobEncoder(b *bytes.Buffer, ut *userTypeInfo, v reflect.Value) {
	// TODO: should we catch panics from the called method?

	var data []byte
	var err error
	// We know it's one of these.
	switch ut.externalEnc {
	case xGob:
		data, err = v.Interface().(GobEncoder).GobEncode()
	case xBinary:
		data, err = v.Interface().(encoding.BinaryMarshaler).MarshalBinary()
	case xText:
		data, err = v.Interface().(encoding.TextMarshaler).MarshalText()
	}
	if err != nil {
		error_(err)
	}
	state := enc.newEncoderState(b)
	state.fieldnum = -1
	state.encodeUint(uint64(len(data)))
	state.b.Write(data)
	enc.freeEncoderState(state)
}

var encOpTable = [...]encOp{
	reflect.Bool:       encBool,
	reflect.Int:        encInt,
	reflect.Int8:       encInt,
	reflect.Int16:      encInt,
	reflect.Int32:      encInt,
	reflect.Int64:      encInt,
	reflect.Uint:       encUint,
	reflect.Uint8:      encUint,
	reflect.Uint16:     encUint,
	reflect.Uint32:     encUint,
	reflect.Uint64:     encUint,
	reflect.Uintptr:    encUint,
	reflect.Float32:    encFloat,
	reflect.Float64:    encFloat,
	reflect.Complex64:  encComplex,
	reflect.Complex128: encComplex,
	reflect.String:     encString,
}

// encOpFor returns (a pointer to) the encoding op for the base type under rt and
// the indirection count to reach it.
func (enc *Encoder) encOpFor(rt reflect.Type, inProgress map[reflect.Type]*encOp) (*encOp, int) {
	ut := userType(rt)
	// If the type implements GobEncoder, we handle it without further processing.
	if ut.externalEnc != 0 {
		return enc.gobEncodeOpFor(ut)
	}
	// If this type is already in progress, it's a recursive type (e.g. map[string]*T).
	// Return the pointer to the op we're already building.
	if opPtr := inProgress[rt]; opPtr != nil {
		return opPtr, ut.indir
	}
	typ := ut.base
	indir := ut.indir
	k := typ.Kind()
	var op encOp
	if int(k) < len(encOpTable) {
		op = encOpTable[k]
	}
	if op == nil {
		inProgress[rt] = &op
		// Special cases
		switch t := typ; t.Kind() {
		case reflect.Slice:
			if t.Elem().Kind() == reflect.Uint8 {
				op = encUint8Array
				break
			}
			// Slices have a header; we decode it to find the underlying array.
			elemOp, elemIndir := enc.encOpFor(t.Elem(), inProgress)
			op = func(i *encInstr, state *encoderState, slice reflect.Value) {
				if !state.sendZero && slice.Len() == 0 {
					return
				}
				state.update(i)
				state.enc.encodeArray(state.b, slice, *elemOp, elemIndir, slice.Len())
			}
		case reflect.Array:
			// True arrays have size in the type.
			elemOp, elemIndir := enc.encOpFor(t.Elem(), inProgress)
			op = func(i *encInstr, state *encoderState, array reflect.Value) {
				state.update(i)
				state.enc.encodeArray(state.b, array, *elemOp, elemIndir, array.Len())
			}
		case reflect.Map:
			keyOp, keyIndir := enc.encOpFor(t.Key(), inProgress)
			elemOp, elemIndir := enc.encOpFor(t.Elem(), inProgress)
			op = func(i *encInstr, state *encoderState, mv reflect.Value) {
				// We send zero-length (but non-nil) maps because the
				// receiver might want to use the map.  (Maps don't use append.)
				if !state.sendZero && mv.IsNil() {
					return
				}
				state.update(i)
				state.enc.encodeMap(state.b, mv, *keyOp, *elemOp, keyIndir, elemIndir)
			}
		case reflect.Struct:
			// Generate a closure that calls out to the engine for the nested type.
			enc.getEncEngine(userType(typ))
			info := mustGetTypeInfo(typ)
			op = func(i *encInstr, state *encoderState, sv reflect.Value) {
				state.update(i)
				// indirect through info to delay evaluation for recursive structs
				state.enc.encodeStruct(state.b, info.encoder, sv)
			}
		case reflect.Interface:
			op = func(i *encInstr, state *encoderState, iv reflect.Value) {
				if !state.sendZero && (!iv.IsValid() || iv.IsNil()) {
					return
				}
				state.update(i)
				state.enc.encodeInterface(state.b, iv)
			}
		}
	}
	if op == nil {
		errorf("can't happen: encode type %s", rt)
	}
	return &op, indir
}

// gobEncodeOpFor returns the op for a type that is known to implement
// GobEncoder.
func (enc *Encoder) gobEncodeOpFor(ut *userTypeInfo) (*encOp, int) {
	rt := ut.user
	if ut.encIndir == -1 {
		rt = reflect.PtrTo(rt)
	} else if ut.encIndir > 0 {
		for i := int8(0); i < ut.encIndir; i++ {
			rt = rt.Elem()
		}
	}
	var op encOp
	op = func(i *encInstr, state *encoderState, v reflect.Value) {
		if ut.encIndir == -1 {
			// Need to climb up one level to turn value into pointer.
			if !v.CanAddr() {
				errorf("unaddressable value of type %s", rt)
			}
			v = v.Addr()
		}
		if !state.sendZero && isZero(v) {
			return
		}
		state.update(i)
		state.enc.encodeGobEncoder(state.b, ut, v)
	}
	return &op, int(ut.encIndir) // encIndir: op will get called with p == address of receiver.
}

// compileEnc returns the engine to compile the type.
func (enc *Encoder) compileEnc(ut *userTypeInfo) *encEngine {
	srt := ut.base
	engine := new(encEngine)
	seen := make(map[reflect.Type]*encOp)
	rt := ut.base
	if ut.externalEnc != 0 {
		rt = ut.user
	}
	if ut.externalEnc == 0 && srt.Kind() == reflect.Struct {
		for fieldNum, wireFieldNum := 0, 0; fieldNum < srt.NumField(); fieldNum++ {
			f := srt.Field(fieldNum)
			if !isSent(&f) {
				continue
			}
			op, indir := enc.encOpFor(f.Type, seen)
			engine.instr = append(engine.instr, encInstr{*op, wireFieldNum, f.Index, indir})
			wireFieldNum++
		}
		if srt.NumField() > 0 && len(engine.instr) == 0 {
			errorf("type %s has no exported fields", rt)
		}
		engine.instr = append(engine.instr, encInstr{encStructTerminator, 0, nil, 0})
	} else {
		engine.instr = make([]encInstr, 1)
		op, indir := enc.encOpFor(rt, seen)
		engine.instr[0] = encInstr{*op, singletonField, nil, indir}
	}
	return engine
}

// getEncEngine returns the engine to compile the type.
// typeLock must be held (or we're in initialization and guaranteed single-threaded).
func (enc *Encoder) getEncEngine(ut *userTypeInfo) *encEngine {
	info, err1 := getTypeInfo(ut)
	if err1 != nil {
		error_(err1)
	}
	if info.encoder == nil {
		// Assign the encEngine now, so recursive types work correctly. But...
		info.encoder = new(encEngine)
		// ... if we fail to complete building the engine, don't cache the half-built machine.
		// Doing this here means we won't cache a type that is itself OK but
		// that contains a nested type that won't compile. The result is consistent
		// error behavior when Encode is called multiple times on the top-level type.
		ok := false
		defer func() {
			if !ok {
				info.encoder = nil
			}
		}()
		info.encoder = enc.compileEnc(ut)
		ok = true
	}
	return info.encoder
}

// lockAndGetEncEngine is a function that locks and compiles.
// This lets us hold the lock only while compiling, not when encoding.
func (enc *Encoder) lockAndGetEncEngine(ut *userTypeInfo) *encEngine {
	typeLock.Lock()
	defer typeLock.Unlock()
	return enc.getEncEngine(ut)
}

func (enc *Encoder) encode(b *bytes.Buffer, value reflect.Value, ut *userTypeInfo) {
	defer catchError(&enc.err)
	engine := enc.lockAndGetEncEngine(ut)
	indir := ut.indir
	if ut.externalEnc != 0 {
		indir = int(ut.encIndir)
	}
	for i := 0; i < indir; i++ {
		value = reflect.Indirect(value)
	}
	if ut.externalEnc == 0 && value.Type().Kind() == reflect.Struct {
		enc.encodeStruct(b, engine, value)
	} else {
		enc.encodeSingle(b, engine, value)
	}
}
