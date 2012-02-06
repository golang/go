// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

// TODO(rsc): When garbage collector changes, revisit
// the allocations in this file that use unsafe.Pointer.

import (
	"bytes"
	"errors"
	"io"
	"math"
	"reflect"
	"unsafe"
)

var (
	errBadUint = errors.New("gob: encoded unsigned integer out of range")
	errBadType = errors.New("gob: unknown type id or corrupted data")
	errRange   = errors.New("gob: bad data: field numbers out of bounds")
)

// decoderState is the execution state of an instance of the decoder. A new state
// is created for nested objects.
type decoderState struct {
	dec *Decoder
	// The buffer is stored with an extra indirection because it may be replaced
	// if we load a type during decode (when reading an interface value).
	b        *bytes.Buffer
	fieldnum int // the last field number read.
	buf      []byte
	next     *decoderState // for free list
}

// We pass the bytes.Buffer separately for easier testing of the infrastructure
// without requiring a full Decoder.
func (dec *Decoder) newDecoderState(buf *bytes.Buffer) *decoderState {
	d := dec.freeList
	if d == nil {
		d = new(decoderState)
		d.dec = dec
		d.buf = make([]byte, uint64Size)
	} else {
		dec.freeList = d.next
	}
	d.b = buf
	return d
}

func (dec *Decoder) freeDecoderState(d *decoderState) {
	d.next = dec.freeList
	dec.freeList = d
}

func overflow(name string) error {
	return errors.New(`value for "` + name + `" out of range`)
}

// decodeUintReader reads an encoded unsigned integer from an io.Reader.
// Used only by the Decoder to read the message length.
func decodeUintReader(r io.Reader, buf []byte) (x uint64, width int, err error) {
	width = 1
	_, err = r.Read(buf[0:width])
	if err != nil {
		return
	}
	b := buf[0]
	if b <= 0x7f {
		return uint64(b), width, nil
	}
	n := -int(int8(b))
	if n > uint64Size {
		err = errBadUint
		return
	}
	width, err = io.ReadFull(r, buf[0:n])
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return
	}
	// Could check that the high byte is zero but it's not worth it.
	for _, b := range buf[0:width] {
		x = x<<8 | uint64(b)
	}
	width++ // +1 for length byte
	return
}

// decodeUint reads an encoded unsigned integer from state.r.
// Does not check for overflow.
func (state *decoderState) decodeUint() (x uint64) {
	b, err := state.b.ReadByte()
	if err != nil {
		error_(err)
	}
	if b <= 0x7f {
		return uint64(b)
	}
	n := -int(int8(b))
	if n > uint64Size {
		error_(errBadUint)
	}
	width, err := state.b.Read(state.buf[0:n])
	if err != nil {
		error_(err)
	}
	// Don't need to check error; it's safe to loop regardless.
	// Could check that the high byte is zero but it's not worth it.
	for _, b := range state.buf[0:width] {
		x = x<<8 | uint64(b)
	}
	return x
}

// decodeInt reads an encoded signed integer from state.r.
// Does not check for overflow.
func (state *decoderState) decodeInt() int64 {
	x := state.decodeUint()
	if x&1 != 0 {
		return ^int64(x >> 1)
	}
	return int64(x >> 1)
}

// decOp is the signature of a decoding operator for a given type.
type decOp func(i *decInstr, state *decoderState, p unsafe.Pointer)

// The 'instructions' of the decoding machine
type decInstr struct {
	op     decOp
	field  int     // field number of the wire type
	indir  int     // how many pointer indirections to reach the value in the struct
	offset uintptr // offset in the structure of the field to encode
	ovfl   error   // error message for overflow/underflow (for arrays, of the elements)
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

// ignoreUint discards a uint value with no destination.
func ignoreUint(i *decInstr, state *decoderState, p unsafe.Pointer) {
	state.decodeUint()
}

// ignoreTwoUints discards a uint value with no destination. It's used to skip
// complex values.
func ignoreTwoUints(i *decInstr, state *decoderState, p unsafe.Pointer) {
	state.decodeUint()
	state.decodeUint()
}

// decBool decodes a uint and stores it as a boolean through p.
func decBool(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(bool))
		}
		p = *(*unsafe.Pointer)(p)
	}
	*(*bool)(p) = state.decodeUint() != 0
}

// decInt8 decodes an integer and stores it as an int8 through p.
func decInt8(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int8))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeInt()
	if v < math.MinInt8 || math.MaxInt8 < v {
		error_(i.ovfl)
	} else {
		*(*int8)(p) = int8(v)
	}
}

// decUint8 decodes an unsigned integer and stores it as a uint8 through p.
func decUint8(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint8))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeUint()
	if math.MaxUint8 < v {
		error_(i.ovfl)
	} else {
		*(*uint8)(p) = uint8(v)
	}
}

// decInt16 decodes an integer and stores it as an int16 through p.
func decInt16(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int16))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeInt()
	if v < math.MinInt16 || math.MaxInt16 < v {
		error_(i.ovfl)
	} else {
		*(*int16)(p) = int16(v)
	}
}

// decUint16 decodes an unsigned integer and stores it as a uint16 through p.
func decUint16(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint16))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeUint()
	if math.MaxUint16 < v {
		error_(i.ovfl)
	} else {
		*(*uint16)(p) = uint16(v)
	}
}

// decInt32 decodes an integer and stores it as an int32 through p.
func decInt32(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int32))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeInt()
	if v < math.MinInt32 || math.MaxInt32 < v {
		error_(i.ovfl)
	} else {
		*(*int32)(p) = int32(v)
	}
}

// decUint32 decodes an unsigned integer and stores it as a uint32 through p.
func decUint32(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(uint32))
		}
		p = *(*unsafe.Pointer)(p)
	}
	v := state.decodeUint()
	if math.MaxUint32 < v {
		error_(i.ovfl)
	} else {
		*(*uint32)(p) = uint32(v)
	}
}

// decInt64 decodes an integer and stores it as an int64 through p.
func decInt64(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(int64))
		}
		p = *(*unsafe.Pointer)(p)
	}
	*(*int64)(p) = int64(state.decodeInt())
}

// decUint64 decodes an unsigned integer and stores it as a uint64 through p.
func decUint64(i *decInstr, state *decoderState, p unsafe.Pointer) {
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

// storeFloat32 decodes an unsigned integer, treats it as a 32-bit floating-point
// number, and stores it through p. It's a helper function for float32 and complex64.
func storeFloat32(i *decInstr, state *decoderState, p unsafe.Pointer) {
	v := floatFromBits(state.decodeUint())
	av := v
	if av < 0 {
		av = -av
	}
	// +Inf is OK in both 32- and 64-bit floats.  Underflow is always OK.
	if math.MaxFloat32 < av && av <= math.MaxFloat64 {
		error_(i.ovfl)
	} else {
		*(*float32)(p) = float32(v)
	}
}

// decFloat32 decodes an unsigned integer, treats it as a 32-bit floating-point
// number, and stores it through p.
func decFloat32(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float32))
		}
		p = *(*unsafe.Pointer)(p)
	}
	storeFloat32(i, state, p)
}

// decFloat64 decodes an unsigned integer, treats it as a 64-bit floating-point
// number, and stores it through p.
func decFloat64(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(float64))
		}
		p = *(*unsafe.Pointer)(p)
	}
	*(*float64)(p) = floatFromBits(uint64(state.decodeUint()))
}

// decComplex64 decodes a pair of unsigned integers, treats them as a
// pair of floating point numbers, and stores them as a complex64 through p.
// The real part comes first.
func decComplex64(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(complex64))
		}
		p = *(*unsafe.Pointer)(p)
	}
	storeFloat32(i, state, p)
	storeFloat32(i, state, unsafe.Pointer(uintptr(p)+unsafe.Sizeof(float32(0))))
}

// decComplex128 decodes a pair of unsigned integers, treats them as a
// pair of floating point numbers, and stores them as a complex128 through p.
// The real part comes first.
func decComplex128(i *decInstr, state *decoderState, p unsafe.Pointer) {
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

// decUint8Slice decodes a byte slice and stores through p a slice header
// describing the data.
// uint8 slices are encoded as an unsigned count followed by the raw bytes.
func decUint8Slice(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new([]uint8))
		}
		p = *(*unsafe.Pointer)(p)
	}
	n := int(state.decodeUint())
	if n < 0 {
		errorf("negative length decoding []byte")
	}
	slice := (*[]uint8)(p)
	if cap(*slice) < n {
		*slice = make([]uint8, n)
	} else {
		*slice = (*slice)[0:n]
	}
	if _, err := state.b.Read(*slice); err != nil {
		errorf("error decoding []byte: %s", err)
	}
}

// decString decodes byte array and stores through p a string header
// describing the data.
// Strings are encoded as an unsigned count followed by the raw bytes.
func decString(i *decInstr, state *decoderState, p unsafe.Pointer) {
	if i.indir > 0 {
		if *(*unsafe.Pointer)(p) == nil {
			*(*unsafe.Pointer)(p) = unsafe.Pointer(new(string))
		}
		p = *(*unsafe.Pointer)(p)
	}
	b := make([]byte, state.decodeUint())
	state.b.Read(b)
	// It would be a shame to do the obvious thing here,
	//	*(*string)(p) = string(b)
	// because we've already allocated the storage and this would
	// allocate again and copy.  So we do this ugly hack, which is even
	// even more unsafe than it looks as it depends the memory
	// representation of a string matching the beginning of the memory
	// representation of a byte slice (a byte slice is longer).
	*(*string)(p) = *(*string)(unsafe.Pointer(&b))
}

// ignoreUint8Array skips over the data for a byte slice value with no destination.
func ignoreUint8Array(i *decInstr, state *decoderState, p unsafe.Pointer) {
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

// decodeSingle decodes a top-level value that is not a struct and stores it through p.
// Such values are preceded by a zero, making them have the memory layout of a
// struct field (although with an illegal field number).
func (dec *Decoder) decodeSingle(engine *decEngine, ut *userTypeInfo, basep uintptr) (err error) {
	state := dec.newDecoderState(&dec.buf)
	state.fieldnum = singletonField
	delta := int(state.decodeUint())
	if delta != 0 {
		errorf("decode: corrupted data: non-zero delta for singleton")
	}
	instr := &engine.instr[singletonField]
	if instr.indir != ut.indir {
		return errors.New("gob: internal error: inconsistent indirection")
	}
	ptr := unsafe.Pointer(basep) // offset will be zero
	if instr.indir > 1 {
		ptr = decIndirect(ptr, instr.indir)
	}
	instr.op(instr, state, ptr)
	dec.freeDecoderState(state)
	return nil
}

// decodeSingle decodes a top-level struct and stores it through p.
// Indir is for the value, not the type.  At the time of the call it may
// differ from ut.indir, which was computed when the engine was built.
// This state cannot arise for decodeSingle, which is called directly
// from the user's value, not from the innards of an engine.
func (dec *Decoder) decodeStruct(engine *decEngine, ut *userTypeInfo, p uintptr, indir int) {
	p = allocate(ut.base, p, indir)
	state := dec.newDecoderState(&dec.buf)
	state.fieldnum = -1
	basep := p
	for state.b.Len() > 0 {
		delta := int(state.decodeUint())
		if delta < 0 {
			errorf("decode: corrupted data: negative delta")
		}
		if delta == 0 { // struct terminator is zero delta fieldnum
			break
		}
		fieldnum := state.fieldnum + delta
		if fieldnum >= len(engine.instr) {
			error_(errRange)
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
	dec.freeDecoderState(state)
}

// ignoreStruct discards the data for a struct with no destination.
func (dec *Decoder) ignoreStruct(engine *decEngine) {
	state := dec.newDecoderState(&dec.buf)
	state.fieldnum = -1
	for state.b.Len() > 0 {
		delta := int(state.decodeUint())
		if delta < 0 {
			errorf("ignore decode: corrupted data: negative delta")
		}
		if delta == 0 { // struct terminator is zero delta fieldnum
			break
		}
		fieldnum := state.fieldnum + delta
		if fieldnum >= len(engine.instr) {
			error_(errRange)
		}
		instr := &engine.instr[fieldnum]
		instr.op(instr, state, unsafe.Pointer(nil))
		state.fieldnum = fieldnum
	}
	dec.freeDecoderState(state)
}

// ignoreSingle discards the data for a top-level non-struct value with no
// destination. It's used when calling Decode with a nil value.
func (dec *Decoder) ignoreSingle(engine *decEngine) {
	state := dec.newDecoderState(&dec.buf)
	state.fieldnum = singletonField
	delta := int(state.decodeUint())
	if delta != 0 {
		errorf("decode: corrupted data: non-zero delta for singleton")
	}
	instr := &engine.instr[singletonField]
	instr.op(instr, state, unsafe.Pointer(nil))
	dec.freeDecoderState(state)
}

// decodeArrayHelper does the work for decoding arrays and slices.
func (dec *Decoder) decodeArrayHelper(state *decoderState, p uintptr, elemOp decOp, elemWid uintptr, length, elemIndir int, ovfl error) {
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

// decodeArray decodes an array and stores it through p, that is, p points to the zeroth element.
// The length is an unsigned integer preceding the elements.  Even though the length is redundant
// (it's part of the type), it's a useful check and is included in the encoding.
func (dec *Decoder) decodeArray(atyp reflect.Type, state *decoderState, p uintptr, elemOp decOp, elemWid uintptr, length, indir, elemIndir int, ovfl error) {
	if indir > 0 {
		p = allocate(atyp, p, 1) // All but the last level has been allocated by dec.Indirect
	}
	if n := state.decodeUint(); n != uint64(length) {
		errorf("length mismatch in decodeArray")
	}
	dec.decodeArrayHelper(state, p, elemOp, elemWid, length, elemIndir, ovfl)
}

// decodeIntoValue is a helper for map decoding.  Since maps are decoded using reflection,
// unlike the other items we can't use a pointer directly.
func decodeIntoValue(state *decoderState, op decOp, indir int, v reflect.Value, ovfl error) reflect.Value {
	instr := &decInstr{op, 0, indir, 0, ovfl}
	up := unsafe.Pointer(unsafeAddr(v))
	if indir > 1 {
		up = decIndirect(up, indir)
	}
	op(instr, state, up)
	return v
}

// decodeMap decodes a map and stores its header through p.
// Maps are encoded as a length followed by key:value pairs.
// Because the internals of maps are not visible to us, we must
// use reflection rather than pointer magic.
func (dec *Decoder) decodeMap(mtyp reflect.Type, state *decoderState, p uintptr, keyOp, elemOp decOp, indir, keyIndir, elemIndir int, ovfl error) {
	if indir > 0 {
		p = allocate(mtyp, p, 1) // All but the last level has been allocated by dec.Indirect
	}
	up := unsafe.Pointer(p)
	if *(*unsafe.Pointer)(up) == nil { // maps are represented as a pointer in the runtime
		// Allocate map.
		*(*unsafe.Pointer)(up) = unsafe.Pointer(reflect.MakeMap(mtyp).Pointer())
	}
	// Maps cannot be accessed by moving addresses around the way
	// that slices etc. can.  We must recover a full reflection value for
	// the iteration.
	v := reflect.ValueOf(unsafe.Unreflect(mtyp, unsafe.Pointer(p)))
	n := int(state.decodeUint())
	for i := 0; i < n; i++ {
		key := decodeIntoValue(state, keyOp, keyIndir, allocValue(mtyp.Key()), ovfl)
		elem := decodeIntoValue(state, elemOp, elemIndir, allocValue(mtyp.Elem()), ovfl)
		v.SetMapIndex(key, elem)
	}
}

// ignoreArrayHelper does the work for discarding arrays and slices.
func (dec *Decoder) ignoreArrayHelper(state *decoderState, elemOp decOp, length int) {
	instr := &decInstr{elemOp, 0, 0, 0, errors.New("no error")}
	for i := 0; i < length; i++ {
		elemOp(instr, state, nil)
	}
}

// ignoreArray discards the data for an array value with no destination.
func (dec *Decoder) ignoreArray(state *decoderState, elemOp decOp, length int) {
	if n := state.decodeUint(); n != uint64(length) {
		errorf("length mismatch in ignoreArray")
	}
	dec.ignoreArrayHelper(state, elemOp, length)
}

// ignoreMap discards the data for a map value with no destination.
func (dec *Decoder) ignoreMap(state *decoderState, keyOp, elemOp decOp) {
	n := int(state.decodeUint())
	keyInstr := &decInstr{keyOp, 0, 0, 0, errors.New("no error")}
	elemInstr := &decInstr{elemOp, 0, 0, 0, errors.New("no error")}
	for i := 0; i < n; i++ {
		keyOp(keyInstr, state, nil)
		elemOp(elemInstr, state, nil)
	}
}

// decodeSlice decodes a slice and stores the slice header through p.
// Slices are encoded as an unsigned length followed by the elements.
func (dec *Decoder) decodeSlice(atyp reflect.Type, state *decoderState, p uintptr, elemOp decOp, elemWid uintptr, indir, elemIndir int, ovfl error) {
	n := int(uintptr(state.decodeUint()))
	if indir > 0 {
		up := unsafe.Pointer(p)
		if *(*unsafe.Pointer)(up) == nil {
			// Allocate the slice header.
			*(*unsafe.Pointer)(up) = unsafe.Pointer(new([]unsafe.Pointer))
		}
		p = *(*uintptr)(up)
	}
	// Allocate storage for the slice elements, that is, the underlying array,
	// if the existing slice does not have the capacity.
	// Always write a header at p.
	hdrp := (*reflect.SliceHeader)(unsafe.Pointer(p))
	if hdrp.Cap < n {
		hdrp.Data = uintptr(unsafe.NewArray(atyp.Elem(), n))
		hdrp.Cap = n
	}
	hdrp.Len = n
	dec.decodeArrayHelper(state, hdrp.Data, elemOp, elemWid, n, elemIndir, ovfl)
}

// ignoreSlice skips over the data for a slice value with no destination.
func (dec *Decoder) ignoreSlice(state *decoderState, elemOp decOp) {
	dec.ignoreArrayHelper(state, elemOp, int(state.decodeUint()))
}

// setInterfaceValue sets an interface value to a concrete value,
// but first it checks that the assignment will succeed.
func setInterfaceValue(ivalue reflect.Value, value reflect.Value) {
	if !value.Type().AssignableTo(ivalue.Type()) {
		errorf("cannot assign value of type %s to %s", value.Type(), ivalue.Type())
	}
	ivalue.Set(value)
}

// decodeInterface decodes an interface value and stores it through p.
// Interfaces are encoded as the name of a concrete type followed by a value.
// If the name is empty, the value is nil and no value is sent.
func (dec *Decoder) decodeInterface(ityp reflect.Type, state *decoderState, p uintptr, indir int) {
	// Create a writable interface reflect.Value.  We need one even for the nil case.
	ivalue := allocValue(ityp)
	// Read the name of the concrete type.
	nr := state.decodeUint()
	if nr < 0 || nr > 1<<31 { // zero is permissible for anonymous types
		errorf("invalid type name length %d", nr)
	}
	b := make([]byte, nr)
	state.b.Read(b)
	name := string(b)
	if name == "" {
		// Copy the representation of the nil interface value to the target.
		// This is horribly unsafe and special.
		*(*[2]uintptr)(unsafe.Pointer(p)) = ivalue.InterfaceData()
		return
	}
	// The concrete type must be registered.
	typ, ok := nameToConcreteType[name]
	if !ok {
		errorf("name not registered for interface: %q", name)
	}
	// Read the type id of the concrete value.
	concreteId := dec.decodeTypeSequence(true)
	if concreteId < 0 {
		error_(dec.err)
	}
	// Byte count of value is next; we don't care what it is (it's there
	// in case we want to ignore the value by skipping it completely).
	state.decodeUint()
	// Read the concrete value.
	value := allocValue(typ)
	dec.decodeValue(concreteId, value)
	if dec.err != nil {
		error_(dec.err)
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
	*(*[2]uintptr)(unsafe.Pointer(p)) = ivalue.InterfaceData()
}

// ignoreInterface discards the data for an interface value with no destination.
func (dec *Decoder) ignoreInterface(state *decoderState) {
	// Read the name of the concrete type.
	b := make([]byte, state.decodeUint())
	_, err := state.b.Read(b)
	if err != nil {
		error_(err)
	}
	id := dec.decodeTypeSequence(true)
	if id < 0 {
		error_(dec.err)
	}
	// At this point, the decoder buffer contains a delimited value. Just toss it.
	state.b.Next(int(state.decodeUint()))
}

// decodeGobDecoder decodes something implementing the GobDecoder interface.
// The data is encoded as a byte slice.
func (dec *Decoder) decodeGobDecoder(state *decoderState, v reflect.Value) {
	// Read the bytes for the value.
	b := make([]byte, state.decodeUint())
	_, err := state.b.Read(b)
	if err != nil {
		error_(err)
	}
	// We know it's a GobDecoder, so just call the method directly.
	err = v.Interface().(GobDecoder).GobDecode(b)
	if err != nil {
		error_(err)
	}
}

// ignoreGobDecoder discards the data for a GobDecoder value with no destination.
func (dec *Decoder) ignoreGobDecoder(state *decoderState) {
	// Read the bytes for the value.
	b := make([]byte, state.decodeUint())
	_, err := state.b.Read(b)
	if err != nil {
		error_(err)
	}
}

// Index by Go types.
var decOpTable = [...]decOp{
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

// decOpFor returns the decoding op for the base type under rt and
// the indirection count to reach it.
func (dec *Decoder) decOpFor(wireId typeId, rt reflect.Type, name string, inProgress map[reflect.Type]*decOp) (*decOp, int) {
	ut := userType(rt)
	// If the type implements GobEncoder, we handle it without further processing.
	if ut.isGobDecoder {
		return dec.gobDecodeOpFor(ut)
	}
	// If this type is already in progress, it's a recursive type (e.g. map[string]*T).
	// Return the pointer to the op we're already building.
	if opPtr := inProgress[rt]; opPtr != nil {
		return opPtr, ut.indir
	}
	typ := ut.base
	indir := ut.indir
	var op decOp
	k := typ.Kind()
	if int(k) < len(decOpTable) {
		op = decOpTable[k]
	}
	if op == nil {
		inProgress[rt] = &op
		// Special cases
		switch t := typ; t.Kind() {
		case reflect.Array:
			name = "element of " + name
			elemId := dec.wireType[wireId].ArrayT.Elem
			elemOp, elemIndir := dec.decOpFor(elemId, t.Elem(), name, inProgress)
			ovfl := overflow(name)
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				state.dec.decodeArray(t, state, uintptr(p), *elemOp, t.Elem().Size(), t.Len(), i.indir, elemIndir, ovfl)
			}

		case reflect.Map:
			name = "element of " + name
			keyId := dec.wireType[wireId].MapT.Key
			elemId := dec.wireType[wireId].MapT.Elem
			keyOp, keyIndir := dec.decOpFor(keyId, t.Key(), name, inProgress)
			elemOp, elemIndir := dec.decOpFor(elemId, t.Elem(), name, inProgress)
			ovfl := overflow(name)
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				up := unsafe.Pointer(p)
				state.dec.decodeMap(t, state, uintptr(up), *keyOp, *elemOp, i.indir, keyIndir, elemIndir, ovfl)
			}

		case reflect.Slice:
			name = "element of " + name
			if t.Elem().Kind() == reflect.Uint8 {
				op = decUint8Slice
				break
			}
			var elemId typeId
			if tt, ok := builtinIdToType[wireId]; ok {
				elemId = tt.(*sliceType).Elem
			} else {
				elemId = dec.wireType[wireId].SliceT.Elem
			}
			elemOp, elemIndir := dec.decOpFor(elemId, t.Elem(), name, inProgress)
			ovfl := overflow(name)
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				state.dec.decodeSlice(t, state, uintptr(p), *elemOp, t.Elem().Size(), i.indir, elemIndir, ovfl)
			}

		case reflect.Struct:
			// Generate a closure that calls out to the engine for the nested type.
			enginePtr, err := dec.getDecEnginePtr(wireId, userType(typ))
			if err != nil {
				error_(err)
			}
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				// indirect through enginePtr to delay evaluation for recursive structs.
				dec.decodeStruct(*enginePtr, userType(typ), uintptr(p), i.indir)
			}
		case reflect.Interface:
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				state.dec.decodeInterface(t, state, uintptr(p), i.indir)
			}
		}
	}
	if op == nil {
		errorf("decode can't handle type %s", rt)
	}
	return &op, indir
}

// decIgnoreOpFor returns the decoding op for a field that has no destination.
func (dec *Decoder) decIgnoreOpFor(wireId typeId) decOp {
	op, ok := decIgnoreOpMap[wireId]
	if !ok {
		if wireId == tInterface {
			// Special case because it's a method: the ignored item might
			// define types and we need to record their state in the decoder.
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				state.dec.ignoreInterface(state)
			}
			return op
		}
		// Special cases
		wire := dec.wireType[wireId]
		switch {
		case wire == nil:
			errorf("bad data: undefined type %s", wireId.string())
		case wire.ArrayT != nil:
			elemId := wire.ArrayT.Elem
			elemOp := dec.decIgnoreOpFor(elemId)
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				state.dec.ignoreArray(state, elemOp, wire.ArrayT.Len)
			}

		case wire.MapT != nil:
			keyId := dec.wireType[wireId].MapT.Key
			elemId := dec.wireType[wireId].MapT.Elem
			keyOp := dec.decIgnoreOpFor(keyId)
			elemOp := dec.decIgnoreOpFor(elemId)
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				state.dec.ignoreMap(state, keyOp, elemOp)
			}

		case wire.SliceT != nil:
			elemId := wire.SliceT.Elem
			elemOp := dec.decIgnoreOpFor(elemId)
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				state.dec.ignoreSlice(state, elemOp)
			}

		case wire.StructT != nil:
			// Generate a closure that calls out to the engine for the nested type.
			enginePtr, err := dec.getIgnoreEnginePtr(wireId)
			if err != nil {
				error_(err)
			}
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				// indirect through enginePtr to delay evaluation for recursive structs
				state.dec.ignoreStruct(*enginePtr)
			}

		case wire.GobEncoderT != nil:
			op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
				state.dec.ignoreGobDecoder(state)
			}
		}
	}
	if op == nil {
		errorf("bad data: ignore can't handle type %s", wireId.string())
	}
	return op
}

// gobDecodeOpFor returns the op for a type that is known to implement
// GobDecoder.
func (dec *Decoder) gobDecodeOpFor(ut *userTypeInfo) (*decOp, int) {
	rcvrType := ut.user
	if ut.decIndir == -1 {
		rcvrType = reflect.PtrTo(rcvrType)
	} else if ut.decIndir > 0 {
		for i := int8(0); i < ut.decIndir; i++ {
			rcvrType = rcvrType.Elem()
		}
	}
	var op decOp
	op = func(i *decInstr, state *decoderState, p unsafe.Pointer) {
		// Caller has gotten us to within one indirection of our value.
		if i.indir > 0 {
			if *(*unsafe.Pointer)(p) == nil {
				*(*unsafe.Pointer)(p) = unsafe.New(ut.base)
			}
		}
		// Now p is a pointer to the base type.  Do we need to climb out to
		// get to the receiver type?
		var v reflect.Value
		if ut.decIndir == -1 {
			v = reflect.ValueOf(unsafe.Unreflect(rcvrType, unsafe.Pointer(&p)))
		} else {
			v = reflect.ValueOf(unsafe.Unreflect(rcvrType, p))
		}
		state.dec.decodeGobDecoder(state, v)
	}
	return &op, int(ut.indir)

}

// compatibleType asks: Are these two gob Types compatible?
// Answers the question for basic types, arrays, maps and slices, plus
// GobEncoder/Decoder pairs.
// Structs are considered ok; fields will be checked later.
func (dec *Decoder) compatibleType(fr reflect.Type, fw typeId, inProgress map[reflect.Type]typeId) bool {
	if rhs, ok := inProgress[fr]; ok {
		return rhs == fw
	}
	inProgress[fr] = fw
	ut := userType(fr)
	wire, ok := dec.wireType[fw]
	// If fr is a GobDecoder, the wire type must be GobEncoder.
	// And if fr is not a GobDecoder, the wire type must not be either.
	if ut.isGobDecoder != (ok && wire.GobEncoderT != nil) { // the parentheses look odd but are correct.
		return false
	}
	if ut.isGobDecoder { // This test trumps all others.
		return true
	}
	switch t := ut.base; t.Kind() {
	default:
		// chan, etc: cannot handle.
		return false
	case reflect.Bool:
		return fw == tBool
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return fw == tInt
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return fw == tUint
	case reflect.Float32, reflect.Float64:
		return fw == tFloat
	case reflect.Complex64, reflect.Complex128:
		return fw == tComplex
	case reflect.String:
		return fw == tString
	case reflect.Interface:
		return fw == tInterface
	case reflect.Array:
		if !ok || wire.ArrayT == nil {
			return false
		}
		array := wire.ArrayT
		return t.Len() == array.Len && dec.compatibleType(t.Elem(), array.Elem, inProgress)
	case reflect.Map:
		if !ok || wire.MapT == nil {
			return false
		}
		MapType := wire.MapT
		return dec.compatibleType(t.Key(), MapType.Key, inProgress) && dec.compatibleType(t.Elem(), MapType.Elem, inProgress)
	case reflect.Slice:
		// Is it an array of bytes?
		if t.Elem().Kind() == reflect.Uint8 {
			return fw == tBytes
		}
		// Extract and compare element types.
		var sw *sliceType
		if tt, ok := builtinIdToType[fw]; ok {
			sw, _ = tt.(*sliceType)
		} else if wire != nil {
			sw = wire.SliceT
		}
		elem := userType(t.Elem()).base
		return sw != nil && dec.compatibleType(elem, sw.Elem, inProgress)
	case reflect.Struct:
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

// compileSingle compiles the decoder engine for a non-struct top-level value, including
// GobDecoders.
func (dec *Decoder) compileSingle(remoteId typeId, ut *userTypeInfo) (engine *decEngine, err error) {
	rt := ut.user
	engine = new(decEngine)
	engine.instr = make([]decInstr, 1) // one item
	name := rt.String()                // best we can do
	if !dec.compatibleType(rt, remoteId, make(map[reflect.Type]typeId)) {
		remoteType := dec.typeString(remoteId)
		// Common confusing case: local interface type, remote concrete type.
		if ut.base.Kind() == reflect.Interface && remoteId != tInterface {
			return nil, errors.New("gob: local interface type " + name + " can only be decoded from remote interface type; received concrete type " + remoteType)
		}
		return nil, errors.New("gob: decoding into local type " + name + ", received remote type " + remoteType)
	}
	op, indir := dec.decOpFor(remoteId, rt, name, make(map[reflect.Type]*decOp))
	ovfl := errors.New(`value for "` + name + `" out of range`)
	engine.instr[singletonField] = decInstr{*op, singletonField, indir, 0, ovfl}
	engine.numInstr = 1
	return
}

// compileIgnoreSingle compiles the decoder engine for a non-struct top-level value that will be discarded.
func (dec *Decoder) compileIgnoreSingle(remoteId typeId) (engine *decEngine, err error) {
	engine = new(decEngine)
	engine.instr = make([]decInstr, 1) // one item
	op := dec.decIgnoreOpFor(remoteId)
	ovfl := overflow(dec.typeString(remoteId))
	engine.instr[0] = decInstr{op, 0, 0, 0, ovfl}
	engine.numInstr = 1
	return
}

// compileDec compiles the decoder engine for a value.  If the value is not a struct,
// it calls out to compileSingle.
func (dec *Decoder) compileDec(remoteId typeId, ut *userTypeInfo) (engine *decEngine, err error) {
	rt := ut.base
	srt := rt
	if srt.Kind() != reflect.Struct ||
		ut.isGobDecoder {
		return dec.compileSingle(remoteId, ut)
	}
	var wireStruct *structType
	// Builtin types can come from global pool; the rest must be defined by the decoder.
	// Also we know we're decoding a struct now, so the client must have sent one.
	if t, ok := builtinIdToType[remoteId]; ok {
		wireStruct, _ = t.(*structType)
	} else {
		wire := dec.wireType[remoteId]
		if wire == nil {
			error_(errBadType)
		}
		wireStruct = wire.StructT
	}
	if wireStruct == nil {
		errorf("type mismatch in decoder: want struct type %s; got non-struct", rt)
	}
	engine = new(decEngine)
	engine.instr = make([]decInstr, len(wireStruct.Field))
	seen := make(map[reflect.Type]*decOp)
	// Loop over the fields of the wire type.
	for fieldnum := 0; fieldnum < len(wireStruct.Field); fieldnum++ {
		wireField := wireStruct.Field[fieldnum]
		if wireField.Name == "" {
			errorf("empty name for remote field of type %s", wireStruct.Name)
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
		if !dec.compatibleType(localField.Type, wireField.Id, make(map[reflect.Type]typeId)) {
			errorf("wrong type (%s) for received field %s.%s", localField.Type, wireStruct.Name, wireField.Name)
		}
		op, indir := dec.decOpFor(wireField.Id, localField.Type, localField.Name, seen)
		engine.instr[fieldnum] = decInstr{*op, fieldnum, indir, uintptr(localField.Offset), ovfl}
		engine.numInstr++
	}
	return
}

// getDecEnginePtr returns the engine for the specified type.
func (dec *Decoder) getDecEnginePtr(remoteId typeId, ut *userTypeInfo) (enginePtr **decEngine, err error) {
	rt := ut.base
	decoderMap, ok := dec.decoderCache[rt]
	if !ok {
		decoderMap = make(map[typeId]**decEngine)
		dec.decoderCache[rt] = decoderMap
	}
	if enginePtr, ok = decoderMap[remoteId]; !ok {
		// To handle recursive types, mark this engine as underway before compiling.
		enginePtr = new(*decEngine)
		decoderMap[remoteId] = enginePtr
		*enginePtr, err = dec.compileDec(remoteId, ut)
		if err != nil {
			delete(decoderMap, remoteId)
		}
	}
	return
}

// emptyStruct is the type we compile into when ignoring a struct value.
type emptyStruct struct{}

var emptyStructType = reflect.TypeOf(emptyStruct{})

// getDecEnginePtr returns the engine for the specified type when the value is to be discarded.
func (dec *Decoder) getIgnoreEnginePtr(wireId typeId) (enginePtr **decEngine, err error) {
	var ok bool
	if enginePtr, ok = dec.ignorerCache[wireId]; !ok {
		// To handle recursive types, mark this engine as underway before compiling.
		enginePtr = new(*decEngine)
		dec.ignorerCache[wireId] = enginePtr
		wire := dec.wireType[wireId]
		if wire != nil && wire.StructT != nil {
			*enginePtr, err = dec.compileDec(wireId, userType(emptyStructType))
		} else {
			*enginePtr, err = dec.compileIgnoreSingle(wireId)
		}
		if err != nil {
			delete(dec.ignorerCache, wireId)
		}
	}
	return
}

// decodeValue decodes the data stream representing a value and stores it in val.
func (dec *Decoder) decodeValue(wireId typeId, val reflect.Value) {
	defer catchError(&dec.err)
	// If the value is nil, it means we should just ignore this item.
	if !val.IsValid() {
		dec.decodeIgnoredValue(wireId)
		return
	}
	// Dereference down to the underlying type.
	ut := userType(val.Type())
	base := ut.base
	var enginePtr **decEngine
	enginePtr, dec.err = dec.getDecEnginePtr(wireId, ut)
	if dec.err != nil {
		return
	}
	engine := *enginePtr
	if st := base; st.Kind() == reflect.Struct && !ut.isGobDecoder {
		if engine.numInstr == 0 && st.NumField() > 0 && len(dec.wireType[wireId].StructT.Field) > 0 {
			name := base.Name()
			errorf("type mismatch: no fields matched compiling decoder for %s", name)
		}
		dec.decodeStruct(engine, ut, uintptr(unsafeAddr(val)), ut.indir)
	} else {
		dec.decodeSingle(engine, ut, uintptr(unsafeAddr(val)))
	}
}

// decodeIgnoredValue decodes the data stream representing a value of the specified type and discards it.
func (dec *Decoder) decodeIgnoredValue(wireId typeId) {
	var enginePtr **decEngine
	enginePtr, dec.err = dec.getIgnoreEnginePtr(wireId)
	if dec.err != nil {
		return
	}
	wire := dec.wireType[wireId]
	if wire != nil && wire.StructT != nil {
		dec.ignoreStruct(*enginePtr)
	} else {
		dec.ignoreSingle(*enginePtr)
	}
}

func init() {
	var iop, uop decOp
	switch reflect.TypeOf(int(0)).Bits() {
	case 32:
		iop = decInt32
		uop = decUint32
	case 64:
		iop = decInt64
		uop = decUint64
	default:
		panic("gob: unknown size of int/uint")
	}
	decOpTable[reflect.Int] = iop
	decOpTable[reflect.Uint] = uop

	// Finally uintptr
	switch reflect.TypeOf(uintptr(0)).Bits() {
	case 32:
		uop = decUint32
	case 64:
		uop = decUint64
	default:
		panic("gob: unknown size of uintptr")
	}
	decOpTable[reflect.Uintptr] = uop
}

// Gob assumes it can call UnsafeAddr on any Value
// in order to get a pointer it can copy data from.
// Values that have just been created and do not point
// into existing structs or slices cannot be addressed,
// so simulate it by returning a pointer to a copy.
// Each call allocates once.
func unsafeAddr(v reflect.Value) uintptr {
	if v.CanAddr() {
		return v.UnsafeAddr()
	}
	x := reflect.New(v.Type()).Elem()
	x.Set(v)
	return x.UnsafeAddr()
}

// Gob depends on being able to take the address
// of zeroed Values it creates, so use this wrapper instead
// of the standard reflect.Zero.
// Each call allocates once.
func allocValue(t reflect.Type) reflect.Value {
	return reflect.New(t).Elem()
}
