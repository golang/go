// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import (
	"bufio";
	"io";
	"log";
	"os";
	"reflect";
)

type byteOrder interface {
	Uint16(b []byte) uint16;
	Uint32(b []byte) uint32;
	Uint64(b []byte) uint64;
	String() string;
}

type olsb struct {}

func (olsb) Uint16(b []byte) uint16 {
	return uint16(b[0]) | uint16(b[1]) << 8;
}

func (olsb) Uint32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1]) << 8 | uint32(b[2]) << 16 | uint32(b[3]) << 24;
}

func (olsb) Uint64(b []byte) uint64 {
	return uint64(b[0]) | uint64(b[1]) << 8 | uint64(b[2]) << 16 | uint64(b[3]) << 24 | uint64(b[4]) << 32 | uint64(b[5]) << 40 | uint64(b[6]) << 48 | uint64(b[7]) << 56;
}

func (olsb) String() string {
	return "LSB";
}

type omsb struct {}

func (omsb) Uint16(b []byte) uint16 {
	return uint16(b[1]) | uint16(b[0]) << 8;
}

func (omsb) Uint32(b []byte) uint32 {
	return uint32(b[3]) | uint32(b[2]) << 8 | uint32(b[1]) << 16 | uint32(b[0]) << 24;
}

func (omsb) Uint64(b []byte) uint64 {
	return uint64(b[7]) | uint64(b[6]) << 8 | uint64(b[5]) << 16 | uint64(b[4]) << 24 | uint64(b[3]) << 32 | uint64(b[2]) << 40 | uint64(b[1]) << 48 | uint64(b[0]) << 56;
}

func (omsb) String() string {
	return "MSB";
}

var (
	lsb = olsb{};
	msb = omsb{};
)

// A binaryReader decodes binary data from another reader.  On an
// error, the Read methods simply return 0 and record the error, to
// make it more convenient to decode long sequences of binary data.
// The caller should use the Error method when convenient to check
// for errors.
type binaryReader struct {
	*bufio.Reader;
	err os.Error;
	order byteOrder;
}

// newBinaryReader creates a new binary data reader backed by the
// given reader and using the given byte order for decoding.
func newBinaryReader(r io.Reader, o byteOrder) *binaryReader {
	return &binaryReader{bufio.NewReader(r), nil, o};
}

// Error returns the recorded error, or nil if no error has occurred.
func (r *binaryReader) Error() os.Error {
	return r.err;
}

func (r *binaryReader) ReadUint8() uint8 {
	var buf [1]byte;
	_, err := io.ReadFull(r.Reader, &buf);
	if r.err == nil && err != nil {
		r.err = err;
	}
	return buf[0];
}

func (r *binaryReader) ReadUint16() uint16 {
	var buf [2]byte;
	_, err := io.ReadFull(r.Reader, &buf);
	if r.err == nil && err != nil {
		r.err = err;
	}
	return r.order.Uint16(&buf);
}

func (r *binaryReader) ReadUint32() uint32 {
	var buf [4]byte;
	_, err := io.ReadFull(r.Reader, &buf);
	if r.err == nil && err != nil {
		r.err = err;
	}
	return r.order.Uint32(&buf);
}

func (r *binaryReader) ReadUint64() uint64 {
	var buf [8]byte;
	_, err := io.ReadFull(r.Reader, &buf);
	if r.err == nil && err != nil {
		r.err = err;
	}
	return r.order.Uint64(&buf);
}

func (r *binaryReader) ReadInt8() int8 {
	return int8(r.ReadUint8());
}

func (r *binaryReader) ReadInt16() int16 {
	return int16(r.ReadUint16());
}

func (r *binaryReader) ReadInt32() int32 {
	return int32(r.ReadUint32());
}

func (r *binaryReader) ReadInt64() int64 {
	return int64(r.ReadUint64());
}

// ReadCString reads a NUL-terminated string.
func (r *binaryReader) ReadCString() string {
	str, err := r.Reader.ReadString('\x00');
	if r.err == nil && err != nil {
		r.err = err;
	}
	n := len(str);
	if n > 0 {
		str = str[0:n-1];
	}
	return str;
}

// ReadValue reads a value according to its reflected type.  This can
// read any of the types for which there is a regular Read method,
// plus structs and arrays.  It assumes structs contain no padding.
func (r *binaryReader) ReadValue(v reflect.Value) {
	switch v := v.(type) {
	case *reflect.ArrayValue:
		l := v.Len();
		for i := 0; i < l; i++ {
			r.ReadValue(v.Elem(i));
		}
	case *reflect.StructValue:
		l := v.NumField();
		for i := 0; i < l; i++ {
			r.ReadValue(v.Field(i));
		}

	case *reflect.Uint8Value:
		v.Set(r.ReadUint8());
	case *reflect.Uint16Value:
		v.Set(r.ReadUint16());
	case *reflect.Uint32Value:
		v.Set(r.ReadUint32());
	case *reflect.Uint64Value:
		v.Set(r.ReadUint64());
	case *reflect.Int8Value:
		v.Set(r.ReadInt8());
	case *reflect.Int16Value:
		v.Set(r.ReadInt16());
	case *reflect.Int32Value:
		v.Set(r.ReadInt32());
	case *reflect.Int64Value:
		v.Set(r.ReadInt64());
	case *reflect.StringValue:
		v.Set(r.ReadCString());

	default:
		log.Crashf("Value of unexpected type %T", v);
	}
}

// ReadAny is a convenience wrapper for ReadValue.  It can be passed a
// pointer any type that can be decoded by ReadValue.
func (r *binaryReader) ReadAny(out interface {}) {
	r.ReadValue(reflect.Indirect(reflect.NewValue(out)));
}
