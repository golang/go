// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

// TODO(rsc): When garbage collector changes, revisit
// the allocations in this file that use unsafe.Pointer.

import (
	"bytes"
	"io"
	"math"
	"os"
	"reflect"
	"unicode"
	"unsafe"
	"utf8"
)

var (
	errBadUint = os.ErrorString("gob: encoded unsigned integer out of range")
	errBadType = os.ErrorString("gob: unknown type id or corrupted data")
	errRange   = os.ErrorString("gob: internal error: field numbers out of bounds")
)

// The execution state of an instance of the decoder. A new state
// is created for nested objects.
type decodeState struct {
	dec *Decoder
	// The buffer is stored with an extra indirection because it may be replaced
	// if we load a type during decode (when reading an interface value).
	b        **bytes.Buffer
	fieldnum int // the last field number read.
	buf      []byte
}

func newDecodeState(dec *Decoder, b **bytes.Buffer) *decodeState {
	d := new(decodeState)
	d.dec = dec
	d.b = b
	d.buf = make([]byte, uint64Size)
	return d
}

func overflow(name string) os.ErrorString {
	return os.ErrorString(`value for "` + name + `" out of range`)
}

// decodeUintReader reads an encoded unsigned integer from an io.Reader.
// Used only by the Decoder to read the message length.
func decodeUintReader(r io.Reader, buf []byte) (x uint64, err os.Error) {
	_, err = r.Read(buf[0:1])
	if err != nil {
		return
	}
	b := buf[0]
	if b <= 0x7f {
		return uint64(b), nil
	}
	nb := -int(int8(b))
	if nb > uint64Size {
		err = errBadUint
		return
	}
	var n int
	n, err = io.ReadFull(r, buf[0:nb])
	if err != nil {
		if err == os.EOF {
			err = io.ErrUnexpectedEOF
		}
		return
	}
	// Could check that the high byte is zero but it's not worth it.
	for i := 0; i < n; i++ {
		x <<= 8
		x |= uint64(buf[i])
	}
	return
}

// decodeUint reads an encoded unsigned integer from state.r.
// Does not check for overflow.
func (state *decodeState) decodeUint() (x uint64) {
	b, err := state.b.ReadByte()
	if err != nil {
		error(err)
	}
	if b <= 0x7f {
		return uint64(b)
	}
	nb := -int(int8(b))
	if nb > uint64Size {
		error(errBadUint)
	}
	n, err := state.b.Read(state.buf[0:nb])
	if err != nil {
		error(err)
	}
	// Don't need to check error; it's safe to loop regardless.
	// Could check that the high byte is zero but it's not worth it.
	for i := 0; i < n; i++ {
		x <<= 8
		x |= uint64(state.buf[i])
	}
	return x
}

// decodeInt reads an encoded signed integer from state.r.
// Does not check for overflow.
func (state *decodeState) decodeInt() int64 {
	x := state.decodeUint()
	if x&1 != 0 {
		return ^int64(x >> 1)
	}
	return int64(x >> 1)
}

type decOp func(i *decInstr, state *decodeState, p unsafe.Pointer)

// The 'instructions' of the decoding machine
type decInstr struct {
	op     decOp
	field  int            // field number of the wire type
	indir  int            // how many pointer indirections to reach the value in the struct
	offset uintptr        // offset in the structure of the field to encode
	ovfl   os.ErrorString // error message for overflow/underflow (for arrays, of the elements)
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
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(unsafe.Pointer))
		}
		p = *(*unsafe.Pointer)(p)
	}
	return p
}

func ignoreUint(i *decInstr, state *decodeState, p unsafe.Pointer) {
	state.decodeUint()
}

func ignoreTwoUints(i *decInstr, state *decodeState, p unsafe.Pointer) {
	state.decodeUint()
	state.decodeUint()
}

func decBool(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(bool))
		}
		p = *(*unsafe.Pointer)(p)
	}
	*(*bool)(p) = state.decodeInt() != 0
}

func decInt8(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int8))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeInt()
	if v < math.MinInt8 || math.MaxInt8 < v {
		error(i.ovfl)
	} else {
		*(*int8)(p) = int8(v)
	}
}

func decUint8(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint8))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeUint()
	if math.MaxUint8 < v {
		error(i.ovfl)
	} else {
		*(*uint8)(p) = uint8(v)
	}
}

func decInt16(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int16))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeInt()
	if v < math.MinInt16 || math.MaxInt16 < v {
		error(i.ovfl)
	} else {
		*(*int16)(p) = int16(v)
	}
}

func decUint16(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint16))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeUint()
	if math.MaxUint16 < v {
		error(i.ovfl)
	} else {
		*(*uint16)(p) = uint16(v)
	}
}

func decInt32(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int32))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeInt()
	if v < math.MinInt32 || math.MaxInt32 < v {
		error(i.ovfl)
	} else {
		*(*int32)(p) = int32(v)
	}
}

func decUint32(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint32))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeUint()
	if math.MaxUint32 < v {
		error(i.ovfl)
	} else {
		*(*uint32)(p) = uint32(v)
	}
}

func decInt64(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int64))
		}
		p = *(*unsafe.Pointer)(p)
	}
	*(*int64)(p) = int64(state.decodeInt())
}

func decUint64(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint64))
		}
		p = *(*unsafe.Pointer)(p)
	}
	*(*uint64)(p) = uint64(state.decodeUint())
}

// Floating-point numbers are transmitted as uint64s holding the bits
// of the underlying representation.  They are sent byte-reversed, with
// the exponent end coming out first, so integer floating point numbers
// (for example) transmit more compactly.  This routine does the
// unswizzling.
func floatFromBits(u uint64) float64 {
	var v uint64
	for i := 0; i < 8; i++ {
		v <<= 8
		v |= u & 0xFF
		u >>= 8
	}
	return math.Float64frombits(v)
}

func storeFloat32(i *decInstr, state *decodeState, p unsafe.Pointer) {
	v := floatFromBits(state.decodeUint())
	av := v
	if av < 0 {
		av = -av
	}
	// +Inf is OK in both 32- and 64-bit floats.  Underflow is always OK.
	if math.MaxFloat32 < av && av <= math.MaxFloat64 {
		error(i.ovfl)
	} else {
		*(*float32)(p) = float32(v)
	}
}

func decFloat32(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float32))
		}
		p = *(*unsafe.Pointer)(p)
	}
	storeFloat32(i, state, p)
}

func decFloat64(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float64))
		}
		p = *(*unsafe.Pointer)(p)
	}
	*(*float64)(p) = floatFromBits(uint64(state.decodeUint()))
}

// Complex numbers are just a pair of floating-point numbers, real part first.
func decComplex64(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(complex64))
		}
		p = *(*unsafe.Pointer)(p)
	}
	storeFloat32(i, state, p)
	storeFloat32(i, state, unsafe.Pointer(uintptr(p)+uintptr(unsafe.Sizeof(float32(0)))))
}

func decComplex128(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(complex128))
		}
		p = *(*unsafe.Pointer)(p)
	}
	real := floatFromBits(uint64(state.decodeUint()))
	imag := floatFromBits(uint64(state.decodeUint()))
	*(*complex128)(p) = complex(real, imag)
}

// uint8 arrays are encoded as an unsigned count followed by the raw bytes.
func decUint8Array(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new([]uint8))
		}
		p = *(*unsafe.Pointer)(p)
	}
	b := make([]uint8, state.decodeUint())
	state.b.Read(b)
	*(*[]uint8)(p) = b
}

// Strings are encoded as an unsigned count followed by the raw bytes.
func decString(i *decInstr, state *decodeState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new([]byte))
		}
		p = *(*unsafe.Pointer)(p)
	}
	b := make([]byte, state.decodeUint())
	state.b.Read(b)
	*(*string)(p) = string(b)
}

func ignoreUint8Array(i *decInstr, state *decodeState, p unsafe.Pointer) {
	b := make([]byte, state.decodeUint())
	state.b.Read(b)
}

// Execution engine

// The encoder engine is an array of instructions indexed by field number of the incoming
// decoder.  It is executed with random access according to field number.
type decEngine struct {
	instr    []decInstr
	numInstr int // the number of active instructions
}

// allocate makes sure storage is available for an object of underlying type rtyp
// that is indir levels of indirection through p.
func allocate(rtyp reflect.Type, p uintptr, indir int) uintptr {
	if indir == 0 {
		return p
	}
	up := unsafe.Pointer(p)
	if indir > 1 {
		up = decIndirect(up, indir)
	}
	if *(*unsafe.Pointer)(up) == nil {
		// Allocate object.
		*(*unsafe.Pointer)(up) = unsafe.New(rtyp)
	}
	return *(*uintptr)(up)
}

func (dec *Decoder) decodeSingle(engine *decEngine, rtyp reflect.Type, b **bytes.Buffer, p uintptr, indir int) (err os.Error) {
	defer catchError(&err)
	p = allocate(rtyp, p, indir)
	state := newDecodeState(dec, b)
	state.fieldnum = singletonField
	basep := p
	delta := int(state.decodeUint())
	if delta != 0 {
		errorf("gob decode: corrupted data: non-zero delta for singleton")
	}
	instr := &engine.instr[singletonField]
	ptr := unsafe.Pointer(basep) // offset will be zero
	if instr.indir > 1 {
		ptr = decIndirect(ptr, instr.indir)
	}
	instr.op(instr, state, ptr)
	return nil
}

func (dec *Decoder) decodeStruct(engine *decEngine, rtyp *reflect.StructType, b **bytes.Buffer, p uintptr, indir int) (err os.Error) {
	defer catchError(&err)
	p = allocate(rtyp, p, indir)
	state := newDecodeState(dec, b)
	state.fieldnum = -1
	basep := p
	for state.b.Len() > 0 {
		delta := int(state.decodeUint())
		if delta < 0 {
			errorf("gob decode: corrupted data: negative delta")
		}
		if delta == 0 { // struct terminator is zero delta fieldnum
			break
		}
		fieldnum := state.fieldnum + delta
		if fieldnum >= len(engine.instr) {
			error(errRange)
			break
		}
		instr := &engine.instr[fieldnum]
		p := unsafe.Pointer(basep + instr.offset)
		if instr.indir > 1 {
			p = decIndirect(p, instr.indir)
		}
		instr.op(instr, state, p)
		state.fieldnum = fieldnum
	}
	return nil
}

func (dec *Decoder) ignoreStruct(engine *decEngine, b **bytes.Buffer) (err os.Error) {
	defer catchError(&err)
	state := newDecodeState(dec, b)
	state.fieldnum = -1
	for state.b.Len() > 0 {
		delta := int(state.decodeUint())
		if delta < 0 {
			errorf("gob ignore decode: corrupted data: negative delta")
		}
		if delta == 0 { // struct terminator is zero delta fieldnum
			break
		}
		fieldnum := state.fieldnum + delta
		if fieldnum >= len(engine.instr) {
			error(errRange)
		}
		instr := &engine.instr[fieldnum]
		instr.op(instr, state, unsafe.Pointer(nil))
		state.fieldnum = fieldnum
	}
	return nil
}

func (dec *Decoder) decodeArrayHelper(state *decodeState, p uintptr, elemOp decOp, elemWid uintptr, length, elemIndir int, ovfl os.ErrorString) {
	instr := &decInstr{elemOp, 0, elemIndir, 0, ovfl}
	for i := 0; i < length; i++ {
		up := unsafe.Pointer(p)
		if elemIndir > 1 {
			up = decIndirect(up, elemIndir)
		}
		elemOp(instr, state, up)
		p += uintptr(elemWid)
	}
}

func (dec *Decoder) decodeArray(atyp *reflect.ArrayType, state *decodeState, p uintptr, elemOp decOp, elemWid uintptr, length, indir, elemIndir int, ovfl os.ErrorString) {
	if indir > 0 {
		p = allocate(atyp, p, 1) // All but the last level has been allocated by dec.Indirect
	}
	if n := state.decodeUint(); n != uint64(length) {
		errorf("gob: length mismatch in decodeArray")
	}
	dec.decodeArrayHelper(state, p, elemOp, elemWid, length, elemIndir, ovfl)
}

func decodeIntoValue(state *decodeState, op decOp, indir int, v reflect.Value, ovfl os.ErrorString) reflect.Value {
	instr := &decInstr{op, 0, indir, 0, ovfl}
	up := unsafe.Pointer(v.Addr())
	if indir > 1 {
		up = decIndirect(up, indir)
	}
	op(instr, state, up)
	return v
}

func (dec *Decoder) decodeMap(mtyp *reflect.MapType, state *decodeState, p uintptr, keyOp, elemOp decOp, indir, keyIndir, elemIndir int, ovfl os.ErrorString) {
	if indir > 0 {
		p = allocate(mtyp, p, 1) // All but the last level has been allocated by dec.Indirect
	}
	up := unsafe.Pointer(p)
	if *(*unsafe.Pointer)(up) == nil { // maps are represented as a pointer in the runtime
		// Allocate map.
		*(*unsafe.Pointer)(up) = unsafe.Pointer(reflect.MakeMap(mtyp).Get())
	}
	// Maps cannot be accessed by moving addresses around the way
	// that slices etc. can.  We must recover a full reflection value for
	// the iteration.
	v := reflect.NewValue(unsafe.Unreflect(mtyp, unsafe.Pointer((p)))).(*reflect.MapValue)
	n := int(state.decodeUint())
	for i := 0; i < n; i++ {
		key := decodeIntoValue(state, keyOp, keyIndir, reflect.MakeZero(mtyp.Key()), ovfl)
		elem := decodeIntoValue(state, elemOp, elemIndir, reflect.MakeZero(mtyp.Elem()), ovfl)
		v.SetElem(key, elem)
	}
}

func (dec *Decoder) ignoreArrayHelper(state *decodeState, elemOp decOp, length int) {
	instr := &decInstr{elemOp, 0, 0, 0, os.ErrorString("no error")}
	for i := 0; i < length; i++ {
		elemOp(instr, state, nil)
	}
}

func (dec *Decoder) ignoreArray(state *decodeState, elemOp decOp, length int) {
	if n := state.decodeUint(); n != uint64(length) {
		errorf("gob: length mismatch in ignoreArray")
	}
	dec.ignoreArrayHelper(state, elemOp, length)
}

func (dec *Decoder) ignoreMap(state *decodeState, keyOp, elemOp decOp) {
	n := int(state.decodeUint())
	keyInstr := &decInstr{keyOp, 0, 0, 0, os.ErrorString("no error")}
	elemInstr := &decInstr{elemOp, 0, 0, 0, os.ErrorString("no error")}
	for i := 0; i < n; i++ {
		keyOp(keyInstr, state, nil)
		elemOp(elemInstr, state, nil)
	}
}

func (dec *Decoder) decodeSlice(atyp *reflect.SliceType, state *decodeState, p uintptr, elemOp decOp, elemWid uintptr, indir, elemIndir int, ovfl os.ErrorString) {
	n := int(uintptr(state.decodeUint()))
	if indir > 0 {
		up := unsafe.Pointer(p)
		if *(*unsafe.Pointer)(up) == nil {
			// Allocate the slice header.
			*(*unsafe.Pointer)(up) = unsafe.Pointer(new([]unsafe.Pointer))
		}
		p = *(*uintptr)(up)
	}
	// Allocate storage for the slice elements, that is, the underlying array.
	// Always write a header at p.
	hdrp := (*reflect.SliceHeader)(unsafe.Pointer(p))
	hdrp.Data = uintptr(unsafe.NewArray(atyp.Elem(), n))
	hdrp.Len = n
	hdrp.Cap = n
	dec.decodeArrayHelper(state, hdrp.Data, elemOp, elemWid, n, elemIndir, ovfl)
}

func (dec *Decoder) ignoreSlice(state *decodeState, elemOp decOp) {
	dec.ignoreArrayHelper(state, elemOp, int(state.decodeUint()))
}

// setInterfaceValue sets an interface value to a concrete value through
// reflection.  If the concrete value does not implement the interface, the
// setting will panic.  This routine turns the panic into an error return.
// This dance avoids manually checking that the value satisfies the
// interface.
// TODO(rsc): avoid panic+recover after fixing issue 327.
func setInterfaceValue(ivalue *reflect.InterfaceValue, value reflect.Value) {
	defer func() {
		if e := recover(); e != nil {
			error(e.(os.Error))
		}
	}()
	ivalue.Set(value)
}

// decodeInterface receives the name of a concrete type followed by its value.
// If the name is empty, the value is nil and no value is sent.
func (dec *Decoder) decodeInterface(ityp *reflect.InterfaceType, state *decodeState, p uintptr, indir int) {
	// Create an interface reflect.Value.  We need one even for the nil case.
	ivalue := reflect.MakeZero(ityp).(*reflect.InterfaceValue)
	// Read the name of the concrete type.
	b := make([]byte, state.decodeUint())
	state.b.Read(b)
	name := string(b)
	if name == "" {
		// Copy the representation of the nil interface value to the target.
		// This is horribly unsafe and special.
		*(*[2]uintptr)(unsafe.Pointer(p)) = ivalue.Get()
		return
	}
	// The concrete type must be registered.
	typ, ok := nameToConcreteType[name]
	if !ok {
		errorf("gob: name not registered for interface: %q", name)
	}
	// Read the concrete value.
	value := reflect.MakeZero(typ)
	dec.decodeValueFromBuffer(value, false, true)
	if dec.err != nil {
		error(dec.err)
	}
	// Allocate the destination interface value.
	if indir > 0 {
		p = allocate(ityp, p, 1) // All but the last level has been allocated by dec.Indirect
	}
	// Assign the concrete value to the interface.
	// Tread carefully; it might not satisfy the interface.
	setInterfaceValue(ivalue, value)
	// Copy the representation of the interface value to the target.
	// This is horribly unsafe and special.
	*(*[2]uintptr)(unsafe.Pointer(p)) = ivalue.Get()
}

func (dec *Decoder) ignoreInterface(state *decodeState) {
	// Read the name of the concrete type.
	b := make([]byte, state.decodeUint())
	_, err := state.b.Read(b)
	if err != nil {
		error(err)
	}
	dec.decodeValueFromBuffer(nil, true, true)
	if dec.err != nil {
		error(err)
	}
}

// Index by Go types.
var decOpMap = []decOp{
	reflect.Bool:       decBool,
	reflect.Int8:       decInt8,
	reflect.Int16:      decInt16,
	reflect.Int32:      decInt32,
	reflect.Int64:      decInt64,
	reflect.Uint8:      decUint8,
	reflect.Uint16:     decUint16,
	reflect.Uint32:     decUint32,
	reflect.Uint64:     decUint64,
	reflect.Float32:    decFloat32,
	reflect.Float64:    decFloat64,
	reflect.Complex64:  decComplex64,
	reflect.Complex128: decComplex128,
	reflect.String:     decString,
}

// Indexed by gob types.  tComplex will be added during type.init().
var decIgnoreOpMap = map[typeId]decOp{
	tBool:    ignoreUint,
	tInt:     ignoreUint,
	tUint:    ignoreUint,
	tFloat:   ignoreUint,
	tBytes:   ignoreUint8Array,
	tString:  ignoreUint8Array,
	tComplex: ignoreTwoUints,
}

// Return the decoding op for the base type under rt and
// the indirection count to reach it.
func (dec *Decoder) decOpFor(wireId typeId, rt reflect.Type, name string) (decOp, int) {
	typ, indir := indirect(rt)
	var op decOp
	k := typ.Kind()
	if int(k) < len(decOpMap) {
		op = decOpMap[k]
	}
	if op == nil {
		// Special cases
		switch t := typ.(type) {
		case *reflect.ArrayType:
			name = "element of " + name
			elemId := dec.wireType[wireId].ArrayT.Elem
			elemOp, elemIndir := dec.decOpFor(elemId, t.Elem(), name)
			ovfl := overflow(name)
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.dec.decodeArray(t, state, uintptr(p), elemOp, t.Elem().Size(), t.Len(), i.indir, elemIndir, ovfl)
			}

		case *reflect.MapType:
			name = "element of " + name
			keyId := dec.wireType[wireId].MapT.Key
			elemId := dec.wireType[wireId].MapT.Elem
			keyOp, keyIndir := dec.decOpFor(keyId, t.Key(), name)
			elemOp, elemIndir := dec.decOpFor(elemId, t.Elem(), name)
			ovfl := overflow(name)
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				up := unsafe.Pointer(p)
				state.dec.decodeMap(t, state, uintptr(up), keyOp, elemOp, i.indir, keyIndir, elemIndir, ovfl)
			}

		case *reflect.SliceType:
			name = "element of " + name
			if t.Elem().Kind() == reflect.Uint8 {
				op = decUint8Array
				break
			}
			var elemId typeId
			if tt, ok := builtinIdToType[wireId]; ok {
				elemId = tt.(*sliceType).Elem
			} else {
				elemId = dec.wireType[wireId].SliceT.Elem
			}
			elemOp, elemIndir := dec.decOpFor(elemId, t.Elem(), name)
			ovfl := overflow(name)
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.dec.decodeSlice(t, state, uintptr(p), elemOp, t.Elem().Size(), i.indir, elemIndir, ovfl)
			}

		case *reflect.StructType:
			// Generate a closure that calls out to the engine for the nested type.
			enginePtr, err := dec.getDecEnginePtr(wireId, typ)
			if err != nil {
				error(err)
			}
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				// indirect through enginePtr to delay evaluation for recursive structs
				err = dec.decodeStruct(*enginePtr, t, state.b, uintptr(p), i.indir)
				if err != nil {
					error(err)
				}
			}
		case *reflect.InterfaceType:
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				dec.decodeInterface(t, state, uintptr(p), i.indir)
			}
		}
	}
	if op == nil {
		errorf("gob: decode can't handle type %s", rt.String())
	}
	return op, indir
}

// Return the decoding op for a field that has no destination.
func (dec *Decoder) decIgnoreOpFor(wireId typeId) decOp {
	op, ok := decIgnoreOpMap[wireId]
	if !ok {
		if wireId == tInterface {
			// Special case because it's a method: the ignored item might
			// define types and we need to record their state in the decoder.
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				dec.ignoreInterface(state)
			}
			return op
		}
		// Special cases
		wire := dec.wireType[wireId]
		switch {
		case wire == nil:
			panic("internal error: can't find ignore op for type " + wireId.string())
		case wire.ArrayT != nil:
			elemId := wire.ArrayT.Elem
			elemOp := dec.decIgnoreOpFor(elemId)
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.dec.ignoreArray(state, elemOp, wire.ArrayT.Len)
			}

		case wire.MapT != nil:
			keyId := dec.wireType[wireId].MapT.Key
			elemId := dec.wireType[wireId].MapT.Elem
			keyOp := dec.decIgnoreOpFor(keyId)
			elemOp := dec.decIgnoreOpFor(elemId)
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.dec.ignoreMap(state, keyOp, elemOp)
			}

		case wire.SliceT != nil:
			elemId := wire.SliceT.Elem
			elemOp := dec.decIgnoreOpFor(elemId)
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				state.dec.ignoreSlice(state, elemOp)
			}

		case wire.StructT != nil:
			// Generate a closure that calls out to the engine for the nested type.
			enginePtr, err := dec.getIgnoreEnginePtr(wireId)
			if err != nil {
				error(err)
			}
			op = func(i *decInstr, state *decodeState, p unsafe.Pointer) {
				// indirect through enginePtr to delay evaluation for recursive structs
				state.dec.ignoreStruct(*enginePtr, state.b)
			}
		}
	}
	if op == nil {
		errorf("ignore can't handle type %s", wireId.string())
	}
	return op
}

// Are these two gob Types compatible?
// Answers the question for basic types, arrays, and slices.
// Structs are considered ok; fields will be checked later.
func (dec *Decoder) compatibleType(fr reflect.Type, fw typeId) bool {
	fr, _ = indirect(fr)
	switch t := fr.(type) {
	default:
		// map, chan, etc: cannot handle.
		return false
	case *reflect.BoolType:
		return fw == tBool
	case *reflect.IntType:
		return fw == tInt
	case *reflect.UintType:
		return fw == tUint
	case *reflect.FloatType:
		return fw == tFloat
	case *reflect.ComplexType:
		return fw == tComplex
	case *reflect.StringType:
		return fw == tString
	case *reflect.InterfaceType:
		return fw == tInterface
	case *reflect.ArrayType:
		wire, ok := dec.wireType[fw]
		if !ok || wire.ArrayT == nil {
			return false
		}
		array := wire.ArrayT
		return t.Len() == array.Len && dec.compatibleType(t.Elem(), array.Elem)
	case *reflect.MapType:
		wire, ok := dec.wireType[fw]
		if !ok || wire.MapT == nil {
			return false
		}
		MapType := wire.MapT
		return dec.compatibleType(t.Key(), MapType.Key) && dec.compatibleType(t.Elem(), MapType.Elem)
	case *reflect.SliceType:
		// Is it an array of bytes?
		if t.Elem().Kind() == reflect.Uint8 {
			return fw == tBytes
		}
		// Extract and compare element types.
		var sw *sliceType
		if tt, ok := builtinIdToType[fw]; ok {
			sw = tt.(*sliceType)
		} else {
			sw = dec.wireType[fw].SliceT
		}
		elem, _ := indirect(t.Elem())
		return sw != nil && dec.compatibleType(elem, sw.Elem)
	case *reflect.StructType:
		return true
	}
	return true
}

// typeString returns a human-readable description of the type identified by remoteId.
func (dec *Decoder) typeString(remoteId typeId) string {
	if t := idToType[remoteId]; t != nil {
		// globally known type.
		return t.string()
	}
	return dec.wireType[remoteId].string()
}


func (dec *Decoder) compileSingle(remoteId typeId, rt reflect.Type) (engine *decEngine, err os.Error) {
	engine = new(decEngine)
	engine.instr = make([]decInstr, 1) // one item
	name := rt.String()                // best we can do
	if !dec.compatibleType(rt, remoteId) {
		return nil, os.ErrorString("gob: wrong type received for local value " + name + ": " + dec.typeString(remoteId))
	}
	op, indir := dec.decOpFor(remoteId, rt, name)
	ovfl := os.ErrorString(`value for "` + name + `" out of range`)
	engine.instr[singletonField] = decInstr{op, singletonField, indir, 0, ovfl}
	engine.numInstr = 1
	return
}

// Is this an exported - upper case - name?
func isExported(name string) bool {
	rune, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(rune)
}

func (dec *Decoder) compileDec(remoteId typeId, rt reflect.Type) (engine *decEngine, err os.Error) {
	defer catchError(&err)
	srt, ok := rt.(*reflect.StructType)
	if !ok {
		return dec.compileSingle(remoteId, rt)
	}
	var wireStruct *structType
	// Builtin types can come from global pool; the rest must be defined by the decoder.
	// Also we know we're decoding a struct now, so the client must have sent one.
	if t, ok := builtinIdToType[remoteId]; ok {
		wireStruct, _ = t.(*structType)
	} else {
		wireStruct = dec.wireType[remoteId].StructT
	}
	if wireStruct == nil {
		errorf("gob: type mismatch in decoder: want struct type %s; got non-struct", rt.String())
	}
	engine = new(decEngine)
	engine.instr = make([]decInstr, len(wireStruct.Field))
	// Loop over the fields of the wire type.
	for fieldnum := 0; fieldnum < len(wireStruct.Field); fieldnum++ {
		wireField := wireStruct.Field[fieldnum]
		if wireField.Name == "" {
			errorf("gob: empty name for remote field of type %s", wireStruct.Name)
		}
		ovfl := overflow(wireField.Name)
		// Find the field of the local type with the same name.
		localField, present := srt.FieldByName(wireField.Name)
		// TODO(r): anonymous names
		if !present || !isExported(wireField.Name) {
			op := dec.decIgnoreOpFor(wireField.Id)
			engine.instr[fieldnum] = decInstr{op, fieldnum, 0, 0, ovfl}
			continue
		}
		if !dec.compatibleType(localField.Type, wireField.Id) {
			errorf("gob: wrong type (%s) for received field %s.%s", localField.Type, wireStruct.Name, wireField.Name)
		}
		op, indir := dec.decOpFor(wireField.Id, localField.Type, localField.Name)
		engine.instr[fieldnum] = decInstr{op, fieldnum, indir, uintptr(localField.Offset), ovfl}
		engine.numInstr++
	}
	return
}

func (dec *Decoder) getDecEnginePtr(remoteId typeId, rt reflect.Type) (enginePtr **decEngine, err os.Error) {
	decoderMap, ok := dec.decoderCache[rt]
	if !ok {
		decoderMap = make(map[typeId]**decEngine)
		dec.decoderCache[rt] = decoderMap
	}
	if enginePtr, ok = decoderMap[remoteId]; !ok {
		// To handle recursive types, mark this engine as underway before compiling.
		enginePtr = new(*decEngine)
		decoderMap[remoteId] = enginePtr
		*enginePtr, err = dec.compileDec(remoteId, rt)
		if err != nil {
			decoderMap[remoteId] = nil, false
		}
	}
	return
}

// When ignoring struct data, in effect we compile it into this type
type emptyStruct struct{}

var emptyStructType = reflect.Typeof(emptyStruct{})

func (dec *Decoder) getIgnoreEnginePtr(wireId typeId) (enginePtr **decEngine, err os.Error) {
	var ok bool
	if enginePtr, ok = dec.ignorerCache[wireId]; !ok {
		// To handle recursive types, mark this engine as underway before compiling.
		enginePtr = new(*decEngine)
		dec.ignorerCache[wireId] = enginePtr
		*enginePtr, err = dec.compileDec(wireId, emptyStructType)
		if err != nil {
			dec.ignorerCache[wireId] = nil, false
		}
	}
	return
}

func (dec *Decoder) decode(wireId typeId, val reflect.Value) os.Error {
	// Dereference down to the underlying struct type.
	rt, indir := indirect(val.Type())
	enginePtr, err := dec.getDecEnginePtr(wireId, rt)
	if err != nil {
		return err
	}
	engine := *enginePtr
	if st, ok := rt.(*reflect.StructType); ok {
		if engine.numInstr == 0 && st.NumField() > 0 && len(dec.wireType[wireId].StructT.Field) > 0 {
			name := rt.Name()
			return os.ErrorString("gob: type mismatch: no fields matched compiling decoder for " + name)
		}
		return dec.decodeStruct(engine, st, dec.state.b, uintptr(val.Addr()), indir)
	}
	return dec.decodeSingle(engine, rt, dec.state.b, uintptr(val.Addr()), indir)
}

func init() {
	var iop, uop decOp
	switch reflect.Typeof(int(0)).Bits() {
	case 32:
		iop = decInt32
		uop = decUint32
	case 64:
		iop = decInt64
		uop = decUint64
	default:
		panic("gob: unknown size of int/uint")
	}
	decOpMap[reflect.Int] = iop
	decOpMap[reflect.Uint] = uop

	// Finally uintptr
	switch reflect.Typeof(uintptr(0)).Bits() {
	case 32:
		uop = decUint32
	case 64:
		uop = decUint64
	default:
		panic("gob: unknown size of uintptr")
	}
	decOpMap[reflect.Uintptr] = uop
}
