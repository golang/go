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
		struct { }	// no field names in common
		struct { c, d int }	// no field names in common

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

	Maps are not supported yet, but they will be.  Interfaces, functions, and channels
	cannot be sent in a gob.  Attempting to encode a value that contains one will
	fail.

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
	example, -129=^128=(^256>>1) encodes as (01 82).

	Floating-point numbers are always sent as a representation of a float64 value.
	That value is converted to a uint64 using math.Float64bits.  The uint64 is then
	byte-reversed and sent as a regular unsigned integer.  The byte-reversal means the
	exponent and high-precision part of the mantissa go first.  Since the low bits are
	often zero, this can save encoding bytes.  For instance, 17.0 is encoded in only
	two bytes (40 e2).

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
	= 7 or (81 87).  Finally, after all the fields have been sent a terminating mark
	denotes the end of the struct.  That mark is a delta=0 value, which has
	representation (80).

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

		bool	1
		int	2
		uint	3
		float	4
		[]byte	5
		string	6
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

import (
	"bytes";
	"io";
	"os";
	"reflect";
	"sync";
)

// An Encoder manages the transmission of type and data information to the
// other side of a connection.
type Encoder struct {
	mutex	sync.Mutex;	// each item must be sent atomically
	w	io.Writer;	// where to send the data
	sent	map[reflect.Type] typeId;	// which types we've already sent
	state	*encoderState;	// so we can encode integers, strings directly
	countState	*encoderState;	// stage for writing counts
	buf	[]byte;	// for collecting the output.
}

// NewEncoder returns a new encoder that will transmit on the io.Writer.
func NewEncoder(w io.Writer) *Encoder {
	enc := new(Encoder);
	enc.w = w;
	enc.sent = make(map[reflect.Type] typeId);
	enc.state = new(encoderState);
	enc.state.b = new(bytes.Buffer);	// the rest isn't important; all we need is buffer and writer
	enc.countState = new(encoderState);
	enc.countState.b = new(bytes.Buffer);	// the rest isn't important; all we need is buffer and writer
	return enc;
}

func (enc *Encoder) badType(rt reflect.Type) {
	enc.state.err = os.ErrorString("gob: can't encode type " + rt.String());
}

// Send the data item preceded by a unsigned count of its length.
func (enc *Encoder) send() {
	// Encode the length.
	encodeUint(enc.countState, uint64(enc.state.b.Len()));
	// Build the buffer.
	countLen := enc.countState.b.Len();
	total := countLen + enc.state.b.Len();
	if total > len(enc.buf) {
		enc.buf = make([]byte, total+1000);	// extra for growth
	}
	// Place the length before the data.
	// TODO(r): avoid the extra copy here.
	enc.countState.b.Read(enc.buf[0:countLen]);
	// Now the data.
	enc.state.b.Read(enc.buf[countLen:total]);
	// Write the data.
	enc.w.Write(enc.buf[0:total]);
}

func (enc *Encoder) sendType(origt reflect.Type) {
	// Drill down to the base type.
	rt, _ := indirect(origt);

	// We only send structs - everything else is basic or an error
	switch rt.(type) {
	default:
		// Basic types do not need to be described.
		return;
	case *reflect.StructType:
		// Structs do need to be described.
		break;
	case *reflect.ChanType, *reflect.FuncType, *reflect.MapType, *reflect.InterfaceType:
		// Probably a bad field in a struct.
		enc.badType(rt);
		return;
	case *reflect.ArrayType, *reflect.SliceType:
		// Array and slice types are not sent, only their element types.
		// If we see one here it's user error; probably a bad top-level value.
		enc.badType(rt);
		return;
	}

	// Have we already sent this type?  This time we ask about the base type.
	if _, alreadySent := enc.sent[rt]; alreadySent {
		return
	}

	// Need to send it.
	typeLock.Lock();
	info, err := getTypeInfo(rt);
	typeLock.Unlock();
	if err != nil {
		enc.state.err = err;
		return;
	}
	// Send the pair (-id, type)
	// Id:
	encodeInt(enc.state, -int64(info.id));
	// Type:
	encode(enc.state.b, info.wire);
	enc.send();

	// Remember we've sent this type.
	enc.sent[rt] = info.id;
	// Remember we've sent the top-level, possibly indirect type too.
	enc.sent[origt] = info.id;
	// Now send the inner types
	st := rt.(*reflect.StructType);
	for i := 0; i < st.NumField(); i++ {
		enc.sendType(st.Field(i).Type);
	}
	return
}

// Encode transmits the data item represented by the empty interface value,
// guaranteeing that all necessary type information has been transmitted first.
func (enc *Encoder) Encode(e interface{}) os.Error {
	if enc.state.b.Len() > 0 || enc.countState.b.Len() > 0 {
		panicln("Encoder: buffer not empty")
	}
	rt, _ := indirect(reflect.Typeof(e));

	// Make sure we're single-threaded through here.
	enc.mutex.Lock();
	defer enc.mutex.Unlock();

	// Make sure the type is known to the other side.
	// First, have we already sent this type?
	if _, alreadySent := enc.sent[rt]; !alreadySent {
		// No, so send it.
		enc.sendType(rt);
		if enc.state.err != nil {
			enc.state.b.Reset();
			enc.countState.b.Reset();
			return enc.state.err
		}
	}

	// Identify the type of this top-level value.
	encodeInt(enc.state, int64(enc.sent[rt]));

	// Encode the object.
	encode(enc.state.b, e);
	enc.send();

	return enc.state.err
}
