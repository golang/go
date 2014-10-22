// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run decgen.go -output dec_helpers.go

package gob

import (
	"encoding"
	"errors"
	"io"
	"math"
	"reflect"
)

var (
	errBadUint = errors.New("gob: encoded unsigned integer out of range")
	errBadType = errors.New("gob: unknown type id or corrupted data")
	errRange   = errors.New("gob: bad data: field numbers out of bounds")
)

type decHelper func(state *decoderState, v reflect.Value, length int, ovfl error) bool

// decoderState is the execution state of an instance of the decoder. A new state
// is created for nested objects.
type decoderState struct {
	dec *Decoder
	// The buffer is stored with an extra indirection because it may be replaced
	// if we load a type during decode (when reading an interface value).
	b        *decBuffer
	fieldnum int // the last field number read.
	buf      []byte
	next     *decoderState // for free list
}

// decBuffer is an extremely simple, fast implementation of a read-only byte buffer.
// It is initialized by calling Size and then copying the data into the slice returned by Bytes().
type decBuffer struct {
	data   []byte
	offset int // Read offset.
}

func (d *decBuffer) Read(p []byte) (int, error) {
	n := copy(p, d.data[d.offset:])
	if n == 0 && len(p) != 0 {
		return 0, io.EOF
	}
	d.offset += n
	return n, nil
}

func (d *decBuffer) Drop(n int) {
	if n > d.Len() {
		panic("drop")
	}
	d.offset += n
}

// Size grows the buffer to exactly n bytes, so d.Bytes() will
// return a slice of length n. Existing data is first discarded.
func (d *decBuffer) Size(n int) {
	d.Reset()
	if cap(d.data) < n {
		d.data = make([]byte, n)
	} else {
		d.data = d.data[0:n]
	}
}

func (d *decBuffer) ReadByte() (byte, error) {
	if d.offset >= len(d.data) {
		return 0, io.EOF
	}
	c := d.data[d.offset]
	d.offset++
	return c, nil
}

func (d *decBuffer) Len() int {
	return len(d.data) - d.offset
}

func (d *decBuffer) Bytes() []byte {
	return d.data[d.offset:]
}

func (d *decBuffer) Reset() {
	d.data = d.data[0:0]
	d.offset = 0
}

// We pass the bytes.Buffer separately for easier testing of the infrastructure
// without requiring a full Decoder.
func (dec *Decoder) newDecoderState(buf *decBuffer) *decoderState {
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
	n, err := io.ReadFull(r, buf[0:width])
	if n == 0 {
		return
	}
	b := buf[0]
	if b <= 0x7f {
		return uint64(b), width, nil
	}
	n = -int(int8(b))
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
type decOp func(i *decInstr, state *decoderState, v reflect.Value)

// The 'instructions' of the decoding machine
type decInstr struct {
	op    decOp
	field int   // field number of the wire type
	index []int // field access indices for destination type
	ovfl  error // error message for overflow/underflow (for arrays, of the elements)
}

// ignoreUint discards a uint value with no destination.
func ignoreUint(i *decInstr, state *decoderState, v reflect.Value) {
	state.decodeUint()
}

// ignoreTwoUints discards a uint value with no destination. It's used to skip
// complex values.
func ignoreTwoUints(i *decInstr, state *decoderState, v reflect.Value) {
	state.decodeUint()
	state.decodeUint()
}

// Since the encoder writes no zeros, if we arrive at a decoder we have
// a value to extract and store.  The field number has already been read
// (it's how we knew to call this decoder).
// Each decoder is responsible for handling any indirections associated
// with the data structure.  If any pointer so reached is nil, allocation must
// be done.

// decAlloc takes a value and returns a settable value that can
// be assigned to. If the value is a pointer, decAlloc guarantees it points to storage.
// The callers to the individual decoders are expected to have used decAlloc.
// The individual decoders don't need to it.
func decAlloc(v reflect.Value) reflect.Value {
	for v.Kind() == reflect.Ptr {
		if v.IsNil() {
			v.Set(reflect.New(v.Type().Elem()))
		}
		v = v.Elem()
	}
	return v
}

// decBool decodes a uint and stores it as a boolean in value.
func decBool(i *decInstr, state *decoderState, value reflect.Value) {
	value.SetBool(state.decodeUint() != 0)
}

// decInt8 decodes an integer and stores it as an int8 in value.
func decInt8(i *decInstr, state *decoderState, value reflect.Value) {
	v := state.decodeInt()
	if v < math.MinInt8 || math.MaxInt8 < v {
		error_(i.ovfl)
	}
	value.SetInt(v)
}

// decUint8 decodes an unsigned integer and stores it as a uint8 in value.
func decUint8(i *decInstr, state *decoderState, value reflect.Value) {
	v := state.decodeUint()
	if math.MaxUint8 < v {
		error_(i.ovfl)
	}
	value.SetUint(v)
}

// decInt16 decodes an integer and stores it as an int16 in value.
func decInt16(i *decInstr, state *decoderState, value reflect.Value) {
	v := state.decodeInt()
	if v < math.MinInt16 || math.MaxInt16 < v {
		error_(i.ovfl)
	}
	value.SetInt(v)
}

// decUint16 decodes an unsigned integer and stores it as a uint16 in value.
func decUint16(i *decInstr, state *decoderState, value reflect.Value) {
	v := state.decodeUint()
	if math.MaxUint16 < v {
		error_(i.ovfl)
	}
	value.SetUint(v)
}

// decInt32 decodes an integer and stores it as an int32 in value.
func decInt32(i *decInstr, state *decoderState, value reflect.Value) {
	v := state.decodeInt()
	if v < math.MinInt32 || math.MaxInt32 < v {
		error_(i.ovfl)
	}
	value.SetInt(v)
}

// decUint32 decodes an unsigned integer and stores it as a uint32 in value.
func decUint32(i *decInstr, state *decoderState, value reflect.Value) {
	v := state.decodeUint()
	if math.MaxUint32 < v {
		error_(i.ovfl)
	}
	value.SetUint(v)
}

// decInt64 decodes an integer and stores it as an int64 in value.
func decInt64(i *decInstr, state *decoderState, value reflect.Value) {
	v := state.decodeInt()
	value.SetInt(v)
}

// decUint64 decodes an unsigned integer and stores it as a uint64 in value.
func decUint64(i *decInstr, state *decoderState, value reflect.Value) {
	v := state.decodeUint()
	value.SetUint(v)
}

// Floating-point numbers are transmitted as uint64s holding the bits
// of the underlying representation.  They are sent byte-reversed, with
// the exponent end coming out first, so integer floating point numbers
// (for example) transmit more compactly.  This routine does the
// unswizzling.
func float64FromBits(u uint64) float64 {
	var v uint64
	for i := 0; i < 8; i++ {
		v <<= 8
		v |= u & 0xFF
		u >>= 8
	}
	return math.Float64frombits(v)
}

// float32FromBits decodes an unsigned integer, treats it as a 32-bit floating-point
// number, and returns it. It's a helper function for float32 and complex64.
// It returns a float64 because that's what reflection needs, but its return
// value is known to be accurately representable in a float32.
func float32FromBits(u uint64, ovfl error) float64 {
	v := float64FromBits(u)
	av := v
	if av < 0 {
		av = -av
	}
	// +Inf is OK in both 32- and 64-bit floats.  Underflow is always OK.
	if math.MaxFloat32 < av && av <= math.MaxFloat64 {
		error_(ovfl)
	}
	return v
}

// decFloat32 decodes an unsigned integer, treats it as a 32-bit floating-point
// number, and stores it in value.
func decFloat32(i *decInstr, state *decoderState, value reflect.Value) {
	value.SetFloat(float32FromBits(state.decodeUint(), i.ovfl))
}

// decFloat64 decodes an unsigned integer, treats it as a 64-bit floating-point
// number, and stores it in value.
func decFloat64(i *decInstr, state *decoderState, value reflect.Value) {
	value.SetFloat(float64FromBits(state.decodeUint()))
}

// decComplex64 decodes a pair of unsigned integers, treats them as a
// pair of floating point numbers, and stores them as a complex64 in value.
// The real part comes first.
func decComplex64(i *decInstr, state *decoderState, value reflect.Value) {
	real := float32FromBits(state.decodeUint(), i.ovfl)
	imag := float32FromBits(state.decodeUint(), i.ovfl)
	value.SetComplex(complex(real, imag))
}

// decComplex128 decodes a pair of unsigned integers, treats them as a
// pair of floating point numbers, and stores them as a complex128 in value.
// The real part comes first.
func decComplex128(i *decInstr, state *decoderState, value reflect.Value) {
	real := float64FromBits(state.decodeUint())
	imag := float64FromBits(state.decodeUint())
	value.SetComplex(complex(real, imag))
}

// decUint8Slice decodes a byte slice and stores in value a slice header
// describing the data.
// uint8 slices are encoded as an unsigned count followed by the raw bytes.
func decUint8Slice(i *decInstr, state *decoderState, value reflect.Value) {
	u := state.decodeUint()
	n := int(u)
	if n < 0 || uint64(n) != u {
		errorf("length of %s exceeds input size (%d bytes)", value.Type(), u)
	}
	if n > state.b.Len() {
		errorf("%s data too long for buffer: %d", value.Type(), n)
	}
	if n > tooBig {
		errorf("byte slice too big: %d", n)
	}
	if value.Cap() < n {
		value.Set(reflect.MakeSlice(value.Type(), n, n))
	} else {
		value.Set(value.Slice(0, n))
	}
	if _, err := state.b.Read(value.Bytes()); err != nil {
		errorf("error decoding []byte: %s", err)
	}
}

// decString decodes byte array and stores in value a string header
// describing the data.
// Strings are encoded as an unsigned count followed by the raw bytes.
func decString(i *decInstr, state *decoderState, value reflect.Value) {
	u := state.decodeUint()
	n := int(u)
	if n < 0 || uint64(n) != u || n > state.b.Len() {
		errorf("length of %s exceeds input size (%d bytes)", value.Type(), u)
	}
	if n > state.b.Len() {
		errorf("%s data too long for buffer: %d", value.Type(), n)
	}
	// Read the data.
	data := make([]byte, n)
	if _, err := state.b.Read(data); err != nil {
		errorf("error decoding string: %s", err)
	}
	value.SetString(string(data))
}

// ignoreUint8Array skips over the data for a byte slice value with no destination.
func ignoreUint8Array(i *decInstr, state *decoderState, value reflect.Value) {
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

// decodeSingle decodes a top-level value that is not a struct and stores it in value.
// Such values are preceded by a zero, making them have the memory layout of a
// struct field (although with an illegal field number).
func (dec *Decoder) decodeSingle(engine *decEngine, ut *userTypeInfo, value reflect.Value) {
	state := dec.newDecoderState(&dec.buf)
	defer dec.freeDecoderState(state)
	state.fieldnum = singletonField
	if state.decodeUint() != 0 {
		errorf("decode: corrupted data: non-zero delta for singleton")
	}
	instr := &engine.instr[singletonField]
	instr.op(instr, state, value)
}

// decodeStruct decodes a top-level struct and stores it in value.
// Indir is for the value, not the type.  At the time of the call it may
// differ from ut.indir, which was computed when the engine was built.
// This state cannot arise for decodeSingle, which is called directly
// from the user's value, not from the innards of an engine.
func (dec *Decoder) decodeStruct(engine *decEngine, ut *userTypeInfo, value reflect.Value) {
	state := dec.newDecoderState(&dec.buf)
	defer dec.freeDecoderState(state)
	state.fieldnum = -1
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
		var field reflect.Value
		if instr.index != nil {
			// Otherwise the field is unknown to us and instr.op is an ignore op.
			field = value.FieldByIndex(instr.index)
			if field.Kind() == reflect.Ptr {
				field = decAlloc(field)
			}
		}
		instr.op(instr, state, field)
		state.fieldnum = fieldnum
	}
}

var noValue reflect.Value

// ignoreStruct discards the data for a struct with no destination.
func (dec *Decoder) ignoreStruct(engine *decEngine) {
	state := dec.newDecoderState(&dec.buf)
	defer dec.freeDecoderState(state)
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
		instr.op(instr, state, noValue)
		state.fieldnum = fieldnum
	}
}

// ignoreSingle discards the data for a top-level non-struct value with no
// destination. It's used when calling Decode with a nil value.
func (dec *Decoder) ignoreSingle(engine *decEngine) {
	state := dec.newDecoderState(&dec.buf)
	defer dec.freeDecoderState(state)
	state.fieldnum = singletonField
	delta := int(state.decodeUint())
	if delta != 0 {
		errorf("decode: corrupted data: non-zero delta for singleton")
	}
	instr := &engine.instr[singletonField]
	instr.op(instr, state, noValue)
}

// decodeArrayHelper does the work for decoding arrays and slices.
func (dec *Decoder) decodeArrayHelper(state *decoderState, value reflect.Value, elemOp decOp, length int, ovfl error, helper decHelper) {
	if helper != nil && helper(state, value, length, ovfl) {
		return
	}
	instr := &decInstr{elemOp, 0, nil, ovfl}
	isPtr := value.Type().Elem().Kind() == reflect.Ptr
	for i := 0; i < length; i++ {
		if state.b.Len() == 0 {
			errorf("decoding array or slice: length exceeds input size (%d elements)", length)
		}
		v := value.Index(i)
		if isPtr {
			v = decAlloc(v)
		}
		elemOp(instr, state, v)
	}
}

// decodeArray decodes an array and stores it in value.
// The length is an unsigned integer preceding the elements.  Even though the length is redundant
// (it's part of the type), it's a useful check and is included in the encoding.
func (dec *Decoder) decodeArray(atyp reflect.Type, state *decoderState, value reflect.Value, elemOp decOp, length int, ovfl error, helper decHelper) {
	if n := state.decodeUint(); n != uint64(length) {
		errorf("length mismatch in decodeArray")
	}
	dec.decodeArrayHelper(state, value, elemOp, length, ovfl, helper)
}

// decodeIntoValue is a helper for map decoding.
func decodeIntoValue(state *decoderState, op decOp, isPtr bool, value reflect.Value, ovfl error) reflect.Value {
	instr := &decInstr{op, 0, nil, ovfl}
	v := value
	if isPtr {
		v = decAlloc(value)
	}
	op(instr, state, v)
	return value
}

// decodeMap decodes a map and stores it in value.
// Maps are encoded as a length followed by key:value pairs.
// Because the internals of maps are not visible to us, we must
// use reflection rather than pointer magic.
func (dec *Decoder) decodeMap(mtyp reflect.Type, state *decoderState, value reflect.Value, keyOp, elemOp decOp, ovfl error) {
	if value.IsNil() {
		// Allocate map.
		value.Set(reflect.MakeMap(mtyp))
	}
	n := int(state.decodeUint())
	keyIsPtr := mtyp.Key().Kind() == reflect.Ptr
	elemIsPtr := mtyp.Elem().Kind() == reflect.Ptr
	for i := 0; i < n; i++ {
		key := decodeIntoValue(state, keyOp, keyIsPtr, allocValue(mtyp.Key()), ovfl)
		elem := decodeIntoValue(state, elemOp, elemIsPtr, allocValue(mtyp.Elem()), ovfl)
		value.SetMapIndex(key, elem)
	}
}

// ignoreArrayHelper does the work for discarding arrays and slices.
func (dec *Decoder) ignoreArrayHelper(state *decoderState, elemOp decOp, length int) {
	instr := &decInstr{elemOp, 0, nil, errors.New("no error")}
	for i := 0; i < length; i++ {
		elemOp(instr, state, noValue)
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
	keyInstr := &decInstr{keyOp, 0, nil, errors.New("no error")}
	elemInstr := &decInstr{elemOp, 0, nil, errors.New("no error")}
	for i := 0; i < n; i++ {
		keyOp(keyInstr, state, noValue)
		elemOp(elemInstr, state, noValue)
	}
}

// decodeSlice decodes a slice and stores it in value.
// Slices are encoded as an unsigned length followed by the elements.
func (dec *Decoder) decodeSlice(state *decoderState, value reflect.Value, elemOp decOp, ovfl error, helper decHelper) {
	u := state.decodeUint()
	typ := value.Type()
	size := uint64(typ.Elem().Size())
	nBytes := u * size
	n := int(u)
	// Take care with overflow in this calculation.
	if n < 0 || uint64(n) != u || nBytes > tooBig || (size > 0 && nBytes/size != u) {
		// We don't check n against buffer length here because if it's a slice
		// of interfaces, there will be buffer reloads.
		errorf("%s slice too big: %d elements of %d bytes", typ.Elem(), u, size)
	}
	if value.Cap() < n {
		value.Set(reflect.MakeSlice(typ, n, n))
	} else {
		value.Set(value.Slice(0, n))
	}
	dec.decodeArrayHelper(state, value, elemOp, n, ovfl, helper)
}

// ignoreSlice skips over the data for a slice value with no destination.
func (dec *Decoder) ignoreSlice(state *decoderState, elemOp decOp) {
	dec.ignoreArrayHelper(state, elemOp, int(state.decodeUint()))
}

// decodeInterface decodes an interface value and stores it in value.
// Interfaces are encoded as the name of a concrete type followed by a value.
// If the name is empty, the value is nil and no value is sent.
func (dec *Decoder) decodeInterface(ityp reflect.Type, state *decoderState, value reflect.Value) {
	// Read the name of the concrete type.
	nr := state.decodeUint()
	if nr < 0 || nr > 1<<31 { // zero is permissible for anonymous types
		errorf("invalid type name length %d", nr)
	}
	if nr > uint64(state.b.Len()) {
		errorf("invalid type name length %d: exceeds input size", nr)
	}
	b := make([]byte, nr)
	state.b.Read(b)
	name := string(b)
	// Allocate the destination interface value.
	if name == "" {
		// Copy the nil interface value to the target.
		value.Set(reflect.Zero(value.Type()))
		return
	}
	if len(name) > 1024 {
		errorf("name too long (%d bytes): %.20q...", len(name), name)
	}
	// The concrete type must be registered.
	registerLock.RLock()
	typ, ok := nameToConcreteType[name]
	registerLock.RUnlock()
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
	v := allocValue(typ)
	dec.decodeValue(concreteId, v)
	if dec.err != nil {
		error_(dec.err)
	}
	// Assign the concrete value to the interface.
	// Tread carefully; it might not satisfy the interface.
	if !typ.AssignableTo(ityp) {
		errorf("%s is not assignable to type %s", typ, ityp)
	}
	// Copy the interface value to the target.
	value.Set(v)
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
	state.b.Drop(int(state.decodeUint()))
}

// decodeGobDecoder decodes something implementing the GobDecoder interface.
// The data is encoded as a byte slice.
func (dec *Decoder) decodeGobDecoder(ut *userTypeInfo, state *decoderState, value reflect.Value) {
	// Read the bytes for the value.
	b := make([]byte, state.decodeUint())
	_, err := state.b.Read(b)
	if err != nil {
		error_(err)
	}
	// We know it's one of these.
	switch ut.externalDec {
	case xGob:
		err = value.Interface().(GobDecoder).GobDecode(b)
	case xBinary:
		err = value.Interface().(encoding.BinaryUnmarshaler).UnmarshalBinary(b)
	case xText:
		err = value.Interface().(encoding.TextUnmarshaler).UnmarshalText(b)
	}
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
func (dec *Decoder) decOpFor(wireId typeId, rt reflect.Type, name string, inProgress map[reflect.Type]*decOp) *decOp {
	ut := userType(rt)
	// If the type implements GobEncoder, we handle it without further processing.
	if ut.externalDec != 0 {
		return dec.gobDecodeOpFor(ut)
	}

	// If this type is already in progress, it's a recursive type (e.g. map[string]*T).
	// Return the pointer to the op we're already building.
	if opPtr := inProgress[rt]; opPtr != nil {
		return opPtr
	}
	typ := ut.base
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
			elemOp := dec.decOpFor(elemId, t.Elem(), name, inProgress)
			ovfl := overflow(name)
			helper := decArrayHelper[t.Elem().Kind()]
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				state.dec.decodeArray(t, state, value, *elemOp, t.Len(), ovfl, helper)
			}

		case reflect.Map:
			keyId := dec.wireType[wireId].MapT.Key
			elemId := dec.wireType[wireId].MapT.Elem
			keyOp := dec.decOpFor(keyId, t.Key(), "key of "+name, inProgress)
			elemOp := dec.decOpFor(elemId, t.Elem(), "element of "+name, inProgress)
			ovfl := overflow(name)
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				state.dec.decodeMap(t, state, value, *keyOp, *elemOp, ovfl)
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
			elemOp := dec.decOpFor(elemId, t.Elem(), name, inProgress)
			ovfl := overflow(name)
			helper := decSliceHelper[t.Elem().Kind()]
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				state.dec.decodeSlice(state, value, *elemOp, ovfl, helper)
			}

		case reflect.Struct:
			// Generate a closure that calls out to the engine for the nested type.
			ut := userType(typ)
			enginePtr, err := dec.getDecEnginePtr(wireId, ut)
			if err != nil {
				error_(err)
			}
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				// indirect through enginePtr to delay evaluation for recursive structs.
				dec.decodeStruct(*enginePtr, ut, value)
			}
		case reflect.Interface:
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				state.dec.decodeInterface(t, state, value)
			}
		}
	}
	if op == nil {
		errorf("decode can't handle type %s", rt)
	}
	return &op
}

// decIgnoreOpFor returns the decoding op for a field that has no destination.
func (dec *Decoder) decIgnoreOpFor(wireId typeId) decOp {
	op, ok := decIgnoreOpMap[wireId]
	if !ok {
		if wireId == tInterface {
			// Special case because it's a method: the ignored item might
			// define types and we need to record their state in the decoder.
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
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
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				state.dec.ignoreArray(state, elemOp, wire.ArrayT.Len)
			}

		case wire.MapT != nil:
			keyId := dec.wireType[wireId].MapT.Key
			elemId := dec.wireType[wireId].MapT.Elem
			keyOp := dec.decIgnoreOpFor(keyId)
			elemOp := dec.decIgnoreOpFor(elemId)
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				state.dec.ignoreMap(state, keyOp, elemOp)
			}

		case wire.SliceT != nil:
			elemId := wire.SliceT.Elem
			elemOp := dec.decIgnoreOpFor(elemId)
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				state.dec.ignoreSlice(state, elemOp)
			}

		case wire.StructT != nil:
			// Generate a closure that calls out to the engine for the nested type.
			enginePtr, err := dec.getIgnoreEnginePtr(wireId)
			if err != nil {
				error_(err)
			}
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
				// indirect through enginePtr to delay evaluation for recursive structs
				state.dec.ignoreStruct(*enginePtr)
			}

		case wire.GobEncoderT != nil, wire.BinaryMarshalerT != nil, wire.TextMarshalerT != nil:
			op = func(i *decInstr, state *decoderState, value reflect.Value) {
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
func (dec *Decoder) gobDecodeOpFor(ut *userTypeInfo) *decOp {
	rcvrType := ut.user
	if ut.decIndir == -1 {
		rcvrType = reflect.PtrTo(rcvrType)
	} else if ut.decIndir > 0 {
		for i := int8(0); i < ut.decIndir; i++ {
			rcvrType = rcvrType.Elem()
		}
	}
	var op decOp
	op = func(i *decInstr, state *decoderState, value reflect.Value) {
		// We now have the base type. We need its address if the receiver is a pointer.
		if value.Kind() != reflect.Ptr && rcvrType.Kind() == reflect.Ptr {
			value = value.Addr()
		}
		state.dec.decodeGobDecoder(ut, state, value)
	}
	return &op
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
	// If wire was encoded with an encoding method, fr must have that method.
	// And if not, it must not.
	// At most one of the booleans in ut is set.
	// We could possibly relax this constraint in the future in order to
	// choose the decoding method using the data in the wireType.
	// The parentheses look odd but are correct.
	if (ut.externalDec == xGob) != (ok && wire.GobEncoderT != nil) ||
		(ut.externalDec == xBinary) != (ok && wire.BinaryMarshalerT != nil) ||
		(ut.externalDec == xText) != (ok && wire.TextMarshalerT != nil) {
		return false
	}
	if ut.externalDec != 0 { // This test trumps all others.
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
	op := dec.decOpFor(remoteId, rt, name, make(map[reflect.Type]*decOp))
	ovfl := errors.New(`value for "` + name + `" out of range`)
	engine.instr[singletonField] = decInstr{*op, singletonField, nil, ovfl}
	engine.numInstr = 1
	return
}

// compileIgnoreSingle compiles the decoder engine for a non-struct top-level value that will be discarded.
func (dec *Decoder) compileIgnoreSingle(remoteId typeId) (engine *decEngine, err error) {
	engine = new(decEngine)
	engine.instr = make([]decInstr, 1) // one item
	op := dec.decIgnoreOpFor(remoteId)
	ovfl := overflow(dec.typeString(remoteId))
	engine.instr[0] = decInstr{op, 0, nil, ovfl}
	engine.numInstr = 1
	return
}

// compileDec compiles the decoder engine for a value.  If the value is not a struct,
// it calls out to compileSingle.
func (dec *Decoder) compileDec(remoteId typeId, ut *userTypeInfo) (engine *decEngine, err error) {
	rt := ut.base
	srt := rt
	if srt.Kind() != reflect.Struct || ut.externalDec != 0 {
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
			engine.instr[fieldnum] = decInstr{op, fieldnum, nil, ovfl}
			continue
		}
		if !dec.compatibleType(localField.Type, wireField.Id, make(map[reflect.Type]typeId)) {
			errorf("wrong type (%s) for received field %s.%s", localField.Type, wireStruct.Name, wireField.Name)
		}
		op := dec.decOpFor(wireField.Id, localField.Type, localField.Name, seen)
		engine.instr[fieldnum] = decInstr{*op, fieldnum, localField.Index, ovfl}
		engine.numInstr++
	}
	return
}

// getDecEnginePtr returns the engine for the specified type.
func (dec *Decoder) getDecEnginePtr(remoteId typeId, ut *userTypeInfo) (enginePtr **decEngine, err error) {
	rt := ut.user
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

// decodeValue decodes the data stream representing a value and stores it in value.
func (dec *Decoder) decodeValue(wireId typeId, value reflect.Value) {
	defer catchError(&dec.err)
	// If the value is nil, it means we should just ignore this item.
	if !value.IsValid() {
		dec.decodeIgnoredValue(wireId)
		return
	}
	// Dereference down to the underlying type.
	ut := userType(value.Type())
	base := ut.base
	var enginePtr **decEngine
	enginePtr, dec.err = dec.getDecEnginePtr(wireId, ut)
	if dec.err != nil {
		return
	}
	value = decAlloc(value)
	engine := *enginePtr
	if st := base; st.Kind() == reflect.Struct && ut.externalDec == 0 {
		if engine.numInstr == 0 && st.NumField() > 0 &&
			dec.wireType[wireId] != nil && len(dec.wireType[wireId].StructT.Field) > 0 {
			name := base.Name()
			errorf("type mismatch: no fields matched compiling decoder for %s", name)
		}
		dec.decodeStruct(engine, ut, value)
	} else {
		dec.decodeSingle(engine, ut, value)
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

// Gob depends on being able to take the address
// of zeroed Values it creates, so use this wrapper instead
// of the standard reflect.Zero.
// Each call allocates once.
func allocValue(t reflect.Type) reflect.Value {
	return reflect.New(t).Elem()
}
