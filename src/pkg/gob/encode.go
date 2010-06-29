// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	The gob package manages streams of gobs - binary values exchanged between an
	Encoder (transmitter) and a Decoder (receiver).  A typical use is transporting
	arguments and results of remote procedure calls (RPCs) such as those provided by
	package "rpc".

	A stream of gobs is self-describing.  Each data item in the stream is preceded by
	a specification of its type, expressed in terms of a small set of predefined
	types.  Pointers are not transmitted, but the things they point to are
	transmitted; that is, the values are flattened.  Recursive types work fine, but
	recursive values (data with cycles) are problematic.  This may change.

	To use gobs, create an Encoder and present it with a series of data items as
	values or addresses that can be dereferenced to values.  (At the moment, these
	items must be structs (struct, *struct, **struct etc.), but this may change.) The
	Encoder makes sure all type information is sent before it is needed.  At the
	receive side, a Decoder retrieves values from the encoded stream and unpacks them
	into local variables.

	The source and destination values/types need not correspond exactly.  For structs,
	fields (identified by name) that are in the source but absent from the receiving
	variable will be ignored.  Fields that are in the receiving variable but missing
	from the transmitted type or value will be ignored in the destination.  If a field
	with the same name is present in both, their types must be compatible. Both the
	receiver and transmitter will do all necessary indirection and dereferencing to
	convert between gobs and actual Go values.  For instance, a gob type that is
	schematically,

		struct { a, b int }

	can be sent from or received into any of these Go types:

		struct { a, b int }	// the same
		*struct { a, b int }	// extra indirection of the struct
		struct { *a, **b int }	// extra indirection of the fields
		struct { a, b int64 }	// different concrete value type; see below

	It may also be received into any of these:

		struct { a, b int }	// the same
		struct { b, a int }	// ordering doesn't matter; matching is by name
		struct { a, b, c int }	// extra field (c) ignored
		struct { b int }	// missing field (a) ignored; data will be dropped
		struct { b, c int }	// missing field (a) ignored; extra field (c) ignored.

	Attempting to receive into these types will draw a decode error:

		struct { a int; b uint }	// change of signedness for b
		struct { a int; b float }	// change of type for b
		struct { }			// no field names in common
		struct { c, d int }		// no field names in common

	Integers are transmitted two ways: arbitrary precision signed integers or
	arbitrary precision unsigned integers.  There is no int8, int16 etc.
	discrimination in the gob format; there are only signed and unsigned integers.  As
	described below, the transmitter sends the value in a variable-length encoding;
	the receiver accepts the value and stores it in the destination variable.
	Floating-point numbers are always sent using IEEE-754 64-bit precision (see
	below).

	Signed integers may be received into any signed integer variable: int, int16, etc.;
	unsigned integers may be received into any unsigned integer variable; and floating
	point values may be received into any floating point variable.  However,
	the destination variable must be able to represent the value or the decode
	operation will fail.

	Structs, arrays and slices are also supported.  Strings and arrays of bytes are
	supported with a special, efficient representation (see below).

	Interfaces, functions, and channels cannot be sent in a gob.  Attempting
	to encode a value that contains one will fail.

	The rest of this comment documents the encoding, details that are not important
	for most users.  Details are presented bottom-up.

	An unsigned integer is sent one of two ways.  If it is less than 128, it is sent
	as a byte with that value.  Otherwise it is sent as a minimal-length big-endian
	(high byte first) byte stream holding the value, preceded by one byte holding the
	byte count, negated.  Thus 0 is transmitted as (00), 7 is transmitted as (07) and
	256 is transmitted as (FE 01 00).

	A boolean is encoded within an unsigned integer: 0 for false, 1 for true.

	A signed integer, i, is encoded within an unsigned integer, u.  Within u, bits 1
	upward contain the value; bit 0 says whether they should be complemented upon
	receipt.  The encode algorithm looks like this:

		uint u;
		if i < 0 {
			u = (^i << 1) | 1	// complement i, bit 0 is 1
		} else {
			u = (i << 1)	// do not complement i, bit 0 is 0
		}
		encodeUnsigned(u)

	The low bit is therefore analogous to a sign bit, but making it the complement bit
	instead guarantees that the largest negative integer is not a special case.  For
	example, -129=^128=(^256>>1) encodes as (FE 01 01).

	Floating-point numbers are always sent as a representation of a float64 value.
	That value is converted to a uint64 using math.Float64bits.  The uint64 is then
	byte-reversed and sent as a regular unsigned integer.  The byte-reversal means the
	exponent and high-precision part of the mantissa go first.  Since the low bits are
	often zero, this can save encoding bytes.  For instance, 17.0 is encoded in only
	three bytes (FE 31 40).

	Strings and slices of bytes are sent as an unsigned count followed by that many
	uninterpreted bytes of the value.

	All other slices and arrays are sent as an unsigned count followed by that many
	elements using the standard gob encoding for their type, recursively.

	Structs are sent as a sequence of (field number, field value) pairs.  The field
	value is sent using the standard gob encoding for its type, recursively.  If a
	field has the zero value for its type, it is omitted from the transmission.  The
	field number is defined by the type of the encoded struct: the first field of the
	encoded type is field 0, the second is field 1, etc.  When encoding a value, the
	field numbers are delta encoded for efficiency and the fields are always sent in
	order of increasing field number; the deltas are therefore unsigned.  The
	initialization for the delta encoding sets the field number to -1, so an unsigned
	integer field 0 with value 7 is transmitted as unsigned delta = 1, unsigned value
	= 7 or (01 0E).  Finally, after all the fields have been sent a terminating mark
	denotes the end of the struct.  That mark is a delta=0 value, which has
	representation (00).

	The representation of types is described below.  When a type is defined on a given
	connection between an Encoder and Decoder, it is assigned a signed integer type
	id.  When Encoder.Encode(v) is called, it makes sure there is an id assigned for
	the type of v and all its elements and then it sends the pair (typeid, encoded-v)
	where typeid is the type id of the encoded type of v and encoded-v is the gob
	encoding of the value v.

	To define a type, the encoder chooses an unused, positive type id and sends the
	pair (-type id, encoded-type) where encoded-type is the gob encoding of a wireType
	description, constructed from these types:

		type wireType struct {
			s	structType;
		}
		type fieldType struct {
			name	string;	// the name of the field.
			id	int;	// the type id of the field, which must be already defined
		}
		type commonType {
			name	string;	// the name of the struct type
			id	int;	// the id of the type, repeated for so it's inside the type
		}
		type structType struct {
			commonType;
			field	[]fieldType;	// the fields of the struct.
		}

	If there are nested type ids, the types for all inner type ids must be defined
	before the top-level type id is used to describe an encoded-v.

	For simplicity in setup, the connection is defined to understand these types a
	priori, as well as the basic gob types int, uint, etc.  Their ids are:

		bool		1
		int		2
		uint		3
		float		4
		[]byte		5
		string		6
		wireType	7
		structType	8
		commonType	9
		fieldType	10

	In summary, a gob stream looks like

		((-type id, encoding of a wireType)* (type id, encoding of a value))*

	where * signifies zero or more repetitions and the type id of a value must
	be predefined or be defined before the value in the stream.
*/
package gob

/*
	For implementers and the curious, here is an encoded example.  Given
		type Point {x, y int}
	and the value
		p := Point{22, 33}
	the bytes transmitted that encode p will be:
		1f ff 81 03 01 01 05 50 6f 69 6e 74 01 ff 82 00 01 02 01 01 78
		01 04 00 01 01 79 01 04 00 00 00 07 ff 82 01 2c 01 42 00 07 ff
		82 01 2c 01 42 00
	They are determined as follows.

	Since this is the first transmission of type Point, the type descriptor
	for Point itself must be sent before the value.  This is the first type
	we've sent on this Encoder, so it has type id 65 (0 through 64 are
	reserved).

		1f	// This item (a type descriptor) is 31 bytes long.
		ff 81	// The negative of the id for the type we're defining, -65.
			// This is one byte (indicated by FF = -1) followed by
			// ^-65<<1 | 1.  The low 1 bit signals to complement the
			// rest upon receipt.

		// Now we send a type descriptor, which is itself a struct (wireType).
		// The type of wireType itself is known (it's built in, as is the type of
		// all its components), so we just need to send a *value* of type wireType
		// that represents type "Point".
		// Here starts the encoding of that value.
		// Set the field number implicitly to zero; this is done at the beginning
		// of every struct, including nested structs.
		03 	// Add 3 to field number; now 3 (wireType.structType; this is a struct).
			// structType starts with an embedded commonType, which appears
			// as a regular structure here too.
		01	// add 1 to field number (now 1); start of embedded commonType.
		01	// add one to field number (now 1, the name of the type)
		05	// string is (unsigned) 5 bytes long
		50 6f 69 6e 74	// wireType.structType.commonType.name = "Point"
		01	// add one to field number (now 2, the id of the type)
		ff 82	// wireType.structType.commonType._id = 65
		00 	// end of embedded wiretype.structType.commonType struct
		01	// add one to field number (now 2, the Field array in wireType.structType)
		02	// There are two fields in the type (len(structType.field))
		01	// Start of first field structure; add 1 to get field number 1: field[0].name
		01	// 1 byte
		78	// structType.field[0].name = "x"
		01	// Add 1 to get field number 2: field[0].id
		04	// structType.field[0].typeId is 2 (signed int).
		00	// End of structType.field[0]; start structType.field[1]; set field number to 0.
		01	// Add 1 to get field number 1: field[1].name
		01	// 1 byte
		79	// structType.field[1].name = "y"
		01	// Add 1 to get field number 2: field[0].id
		04	// struct.Type.field[1].typeId is 2 (signed int).
		00	// End of structType.field[1]; end of structType.field.
		00	// end of wireType.structType structure
		00	// end of wireType structure

	Now we can send the Point value.  Again the field number resets to zero:

		07 // this value is 7 bytes long
		ff 82 // the type number, 65 (1 byte (-FF) followed by 65<<1)
		01 // add one to field number, yielding field 1
		2c // encoding of signed "22" (0x22 = 44 = 22<<1); Point.x = 22
		01 // add one to field number, yielding field 2
		42 // encoding of signed "33" (0x42 = 66 = 33<<1); Point.y = 33
		00 // end of structure

	The type encoding is long and fairly intricate but we send it only once.
	If p is transmitted a second time, the type is already known so the
	output will be just:

		07 ff 82 01 2c 01 42 00
*/

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
	b        *bytes.Buffer
	err      os.Error             // error encountered during encoding.
	sendZero bool                 // encoding an array element or map key/value pair; send zero values
	fieldnum int                  // the last field number written.
	buf      [1 + uint64Size]byte // buffer used by the encoder; here to avoid allocation.
}

// Unsigned integers have a two-state encoding.  If the number is less
// than 128 (0 through 0x7F), its value is written directly.
// Otherwise the value is written in big-endian byte order preceded
// by the byte length, negated.

// encodeUint writes an encoded unsigned integer to state.b.  Sets state.err.
// If state.err is already non-nil, it does nothing.
func encodeUint(state *encoderState, x uint64) {
	if state.err != nil {
		return
	}
	if x <= 0x7F {
		state.err = state.b.WriteByte(uint8(x))
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
	n, state.err = state.b.Write(state.buf[m : uint64Size+1])
}

// encodeInt writes an encoded signed integer to state.w.
// The low bit of the encoding says whether to bit complement the (other bits of the) uint to recover the int.
// Sets state.err. If state.err is already non-nil, it does nothing.
func encodeInt(state *encoderState, i int64) {
	var x uint64
	if i < 0 {
		x = uint64(^i<<1) | 1
	} else {
		x = uint64(i << 1)
	}
	encodeUint(state, uint64(x))
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
		encodeUint(state, uint64(instr.field-state.fieldnum))
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
			encodeUint(state, 1)
		} else {
			encodeUint(state, 0)
		}
	}
}

func encInt(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeInt(state, v)
	}
}

func encUint(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeUint(state, v)
	}
}

func encInt8(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int8)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeInt(state, v)
	}
}

func encUint8(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint8)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeUint(state, v)
	}
}

func encInt16(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int16)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeInt(state, v)
	}
}

func encUint16(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint16)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeUint(state, v)
	}
}

func encInt32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := int64(*(*int32)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeInt(state, v)
	}
}

func encUint32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uint32)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeUint(state, v)
	}
}

func encInt64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := *(*int64)(p)
	if v != 0 || state.sendZero {
		state.update(i)
		encodeInt(state, v)
	}
}

func encUint64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := *(*uint64)(p)
	if v != 0 || state.sendZero {
		state.update(i)
		encodeUint(state, v)
	}
}

func encUintptr(i *encInstr, state *encoderState, p unsafe.Pointer) {
	v := uint64(*(*uintptr)(p))
	if v != 0 || state.sendZero {
		state.update(i)
		encodeUint(state, v)
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

func encFloat(i *encInstr, state *encoderState, p unsafe.Pointer) {
	f := *(*float)(p)
	if f != 0 || state.sendZero {
		v := floatBits(float64(f))
		state.update(i)
		encodeUint(state, v)
	}
}

func encFloat32(i *encInstr, state *encoderState, p unsafe.Pointer) {
	f := *(*float32)(p)
	if f != 0 || state.sendZero {
		v := floatBits(float64(f))
		state.update(i)
		encodeUint(state, v)
	}
}

func encFloat64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	f := *(*float64)(p)
	if f != 0 || state.sendZero {
		state.update(i)
		v := floatBits(f)
		encodeUint(state, v)
	}
}

// Complex numbers are just a pair of floating-point numbers, real part first.
func encComplex(i *encInstr, state *encoderState, p unsafe.Pointer) {
	c := *(*complex)(p)
	if c != 0+0i || state.sendZero {
		rpart := floatBits(float64(real(c)))
		ipart := floatBits(float64(imag(c)))
		state.update(i)
		encodeUint(state, rpart)
		encodeUint(state, ipart)
	}
}

func encComplex64(i *encInstr, state *encoderState, p unsafe.Pointer) {
	c := *(*complex64)(p)
	if c != 0+0i || state.sendZero {
		rpart := floatBits(float64(real(c)))
		ipart := floatBits(float64(imag(c)))
		state.update(i)
		encodeUint(state, rpart)
		encodeUint(state, ipart)
	}
}

func encComplex128(i *encInstr, state *encoderState, p unsafe.Pointer) {
	c := *(*complex128)(p)
	if c != 0+0i || state.sendZero {
		rpart := floatBits(real(c))
		ipart := floatBits(imag(c))
		state.update(i)
		encodeUint(state, rpart)
		encodeUint(state, ipart)
	}
}

// Byte arrays are encoded as an unsigned count followed by the raw bytes.
func encUint8Array(i *encInstr, state *encoderState, p unsafe.Pointer) {
	b := *(*[]byte)(p)
	if len(b) > 0 || state.sendZero {
		state.update(i)
		encodeUint(state, uint64(len(b)))
		state.b.Write(b)
	}
}

// Strings are encoded as an unsigned count followed by the raw bytes.
func encString(i *encInstr, state *encoderState, p unsafe.Pointer) {
	s := *(*string)(p)
	if len(s) > 0 || state.sendZero {
		state.update(i)
		encodeUint(state, uint64(len(s)))
		io.WriteString(state.b, s)
	}
}

// The end of a struct is marked by a delta field number of 0.
func encStructTerminator(i *encInstr, state *encoderState, p unsafe.Pointer) {
	encodeUint(state, 0)
}

// Execution engine

// The encoder engine is an array of instructions indexed by field number of the encoding
// data, typically a struct.  It is executed top to bottom, walking the struct.
type encEngine struct {
	instr []encInstr
}

const singletonField = 0

func encodeSingle(engine *encEngine, b *bytes.Buffer, basep uintptr) os.Error {
	state := new(encoderState)
	state.b = b
	state.fieldnum = singletonField
	// There is no surrounding struct to frame the transmission, so we must
	// generate data even if the item is zero.  To do this, set sendZero.
	state.sendZero = true
	instr := &engine.instr[singletonField]
	p := unsafe.Pointer(basep) // offset will be zero
	if instr.indir > 0 {
		if p = encIndirect(p, instr.indir); p == nil {
			return nil
		}
	}
	instr.op(instr, state, p)
	return state.err
}

func encodeStruct(engine *encEngine, b *bytes.Buffer, basep uintptr) os.Error {
	state := new(encoderState)
	state.b = b
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
		if state.err != nil {
			break
		}
	}
	return state.err
}

func encodeArray(b *bytes.Buffer, p uintptr, op encOp, elemWid uintptr, elemIndir int, length int) os.Error {
	state := new(encoderState)
	state.b = b
	state.fieldnum = -1
	state.sendZero = true
	encodeUint(state, uint64(length))
	for i := 0; i < length && state.err == nil; i++ {
		elemp := p
		up := unsafe.Pointer(elemp)
		if elemIndir > 0 {
			if up = encIndirect(up, elemIndir); up == nil {
				state.err = os.ErrorString("gob: encodeArray: nil element")
				break
			}
			elemp = uintptr(up)
		}
		op(nil, state, unsafe.Pointer(elemp))
		p += uintptr(elemWid)
	}
	return state.err
}

func encodeReflectValue(state *encoderState, v reflect.Value, op encOp, indir int) {
	for i := 0; i < indir && v != nil; i++ {
		v = reflect.Indirect(v)
	}
	if v == nil {
		state.err = os.ErrorString("gob: encodeReflectValue: nil element")
		return
	}
	op(nil, state, unsafe.Pointer(v.Addr()))
}

func encodeMap(b *bytes.Buffer, mv *reflect.MapValue, keyOp, elemOp encOp, keyIndir, elemIndir int) os.Error {
	state := new(encoderState)
	state.b = b
	state.fieldnum = -1
	state.sendZero = true
	keys := mv.Keys()
	encodeUint(state, uint64(len(keys)))
	for _, key := range keys {
		if state.err != nil {
			break
		}
		encodeReflectValue(state, key, keyOp, keyIndir)
		encodeReflectValue(state, mv.Elem(key), elemOp, elemIndir)
	}
	return state.err
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
	reflect.Float:      encFloat,
	reflect.Float32:    encFloat32,
	reflect.Float64:    encFloat64,
	reflect.Complex:    encComplex,
	reflect.Complex64:  encComplex64,
	reflect.Complex128: encComplex128,
	reflect.String:     encString,
}

// Return the encoding op for the base type under rt and
// the indirection count to reach it.
func encOpFor(rt reflect.Type) (encOp, int, os.Error) {
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
			elemOp, indir, err := encOpFor(t.Elem())
			if err != nil {
				return nil, 0, err
			}
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				slice := (*reflect.SliceHeader)(p)
				if slice.Len == 0 {
					return
				}
				state.update(i)
				state.err = encodeArray(state.b, slice.Data, elemOp, t.Elem().Size(), indir, int(slice.Len))
			}
		case *reflect.ArrayType:
			// True arrays have size in the type.
			elemOp, indir, err := encOpFor(t.Elem())
			if err != nil {
				return nil, 0, err
			}
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				slice := (*reflect.SliceHeader)(p)
				if slice.Len == 0 {
					return
				}
				state.update(i)
				state.err = encodeArray(state.b, uintptr(p), elemOp, t.Elem().Size(), indir, t.Len())
			}
		case *reflect.MapType:
			keyOp, keyIndir, err := encOpFor(t.Key())
			if err != nil {
				return nil, 0, err
			}
			elemOp, elemIndir, err := encOpFor(t.Elem())
			if err != nil {
				return nil, 0, err
			}
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				// Maps cannot be accessed by moving addresses around the way
				// that slices etc. can.  We must recover a full reflection value for
				// the iteration.
				v := reflect.NewValue(unsafe.Unreflect(t, unsafe.Pointer((p))))
				mv := reflect.Indirect(v).(*reflect.MapValue)
				if mv.Len() == 0 {
					return
				}
				state.update(i)
				state.err = encodeMap(state.b, mv, keyOp, elemOp, keyIndir, elemIndir)
			}
		case *reflect.StructType:
			// Generate a closure that calls out to the engine for the nested type.
			_, err := getEncEngine(typ)
			if err != nil {
				return nil, 0, err
			}
			info := mustGetTypeInfo(typ)
			op = func(i *encInstr, state *encoderState, p unsafe.Pointer) {
				state.update(i)
				// indirect through info to delay evaluation for recursive structs
				state.err = encodeStruct(info.encoder, state.b, uintptr(p))
			}
		}
	}
	if op == nil {
		return op, indir, os.ErrorString("gob enc: can't happen: encode type " + rt.String())
	}
	return op, indir, nil
}

// The local Type was compiled from the actual value, so we know it's compatible.
func compileEnc(rt reflect.Type) (*encEngine, os.Error) {
	srt, isStruct := rt.(*reflect.StructType)
	engine := new(encEngine)
	if isStruct {
		engine.instr = make([]encInstr, srt.NumField()+1) // +1 for terminator
		for fieldnum := 0; fieldnum < srt.NumField(); fieldnum++ {
			f := srt.Field(fieldnum)
			op, indir, err := encOpFor(f.Type)
			if err != nil {
				return nil, err
			}
			engine.instr[fieldnum] = encInstr{op, fieldnum, indir, uintptr(f.Offset)}
		}
		engine.instr[srt.NumField()] = encInstr{encStructTerminator, 0, 0, 0}
	} else {
		engine.instr = make([]encInstr, 1)
		op, indir, err := encOpFor(rt)
		if err != nil {
			return nil, err
		}
		engine.instr[0] = encInstr{op, singletonField, indir, 0} // offset is zero
	}
	return engine, nil
}

// typeLock must be held (or we're in initialization and guaranteed single-threaded).
// The reflection type must have all its indirections processed out.
func getEncEngine(rt reflect.Type) (*encEngine, os.Error) {
	info, err := getTypeInfo(rt)
	if err != nil {
		return nil, err
	}
	if info.encoder == nil {
		// mark this engine as underway before compiling to handle recursive types.
		info.encoder = new(encEngine)
		info.encoder, err = compileEnc(rt)
	}
	return info.encoder, err
}

func encode(b *bytes.Buffer, value reflect.Value) os.Error {
	// Dereference down to the underlying object.
	rt, indir := indirect(value.Type())
	for i := 0; i < indir; i++ {
		value = reflect.Indirect(value)
	}
	typeLock.Lock()
	engine, err := getEncEngine(rt)
	typeLock.Unlock()
	if err != nil {
		return err
	}
	if _, ok := value.(*reflect.StructValue); ok {
		return encodeStruct(engine, b, value.Addr())
	}
	return encodeSingle(engine, b, value.Addr())
}
