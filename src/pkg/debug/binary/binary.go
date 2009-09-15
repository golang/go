// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements translation between
// unsigned integer values and byte sequences.
package binary

import (
	"math";
	"io";
	"os";
	"reflect";
)

// A ByteOrder specifies how to convert byte sequences into
// 16-, 32-, or 64-bit unsigned integers.
type ByteOrder interface {
	Uint16(b []byte) uint16;
	Uint32(b []byte) uint32;
	Uint64(b []byte) uint64;
	String() string;
}

// This is byte instead of struct{} so that it can be compared,
// allowing, e.g., order == binary.LittleEndian.
type unused byte

var LittleEndian ByteOrder = littleEndian(0)
var BigEndian ByteOrder = bigEndian(0)

type littleEndian unused

func (littleEndian) Uint16(b []byte) uint16 {
	return uint16(b[0]) | uint16(b[1])<<8;
}

func (littleEndian) Uint32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24;
}

func (littleEndian) Uint64(b []byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56;
}

func (littleEndian) String() string {
	return "LittleEndian";
}

func (littleEndian) GoString() string {
	return "binary.LittleEndian";
}

type bigEndian unused

func (bigEndian) Uint16(b []byte) uint16 {
	return uint16(b[1]) | uint16(b[0])<<8;
}

func (bigEndian) Uint32(b []byte) uint32 {
	return uint32(b[3]) | uint32(b[2])<<8 | uint32(b[1])<<16 | uint32(b[0])<<24;
}

func (bigEndian) Uint64(b []byte) uint64 {
	return uint64(b[7]) | uint64(b[6])<<8 | uint64(b[5])<<16 | uint64(b[4])<<24 |
		uint64(b[3])<<32 | uint64(b[2])<<40 | uint64(b[1])<<48 | uint64(b[0])<<56;
}

func (bigEndian) String() string {
	return "BigEndian";
}

func (bigEndian) GoString() string {
	return "binary.BigEndian";
}

// Read reads structured binary data from r into data.
// Data must be a pointer to a fixed-size value.
// A fixed-size value is either a fixed-size integer
// (int8, uint8, int16, uint16, ...) or an array or struct
// containing only fixed-size values.  Bytes read from
// r are decoded using order and written to successive
// fields of the data.
func Read(r io.Reader, order ByteOrder, data interface{}) os.Error {
	v := reflect.NewValue(data).(*reflect.PtrValue).Elem();
	size := sizeof(v.Type());
	if size < 0 {
		return os.NewError("binary.Read: invalid type " + v.Type().String());
	}
	d := &decoder{order: order, buf: make([]byte, size)};
	if _, err := io.ReadFull(r, d.buf); err != nil {
		return err;
	}
	d.value(v);
	return nil;
}

func sizeof(t reflect.Type) int {
	switch t := t.(type) {
	case *reflect.ArrayType:
		n := sizeof(t.Elem());
		if n < 0 {
			return -1;
		}
		return t.Len() * n;

	case *reflect.StructType:
		sum := 0;
		for i, n := 0, t.NumField(); i < n; i++ {
			s := sizeof(t.Field(i).Type);
			if s < 0 {
				return -1;
			}
			sum += s;
		}
		return sum;

	case *reflect.Uint8Type:
		return 1;
	case *reflect.Uint16Type:
		return 2;
	case *reflect.Uint32Type:
		return 4;
	case *reflect.Uint64Type:
		return 8;
	case *reflect.Int8Type:
		return 1;
	case *reflect.Int16Type:
		return 2;
	case *reflect.Int32Type:
		return 4;
	case *reflect.Int64Type:
		return 8;
	case *reflect.Float32Type:
		return 4;
	case *reflect.Float64Type:
		return 8;
	}
	return -1;
}

type decoder struct {
	order ByteOrder;
	buf []byte;
}

func (d *decoder) uint8() uint8 {
	x := d.buf[0];
	d.buf = d.buf[1:len(d.buf)];
	return x;
}

func (d *decoder) uint16() uint16 {
	x := d.order.Uint16(d.buf[0:2]);
	d.buf = d.buf[2:len(d.buf)];
	return x;
}

func (d *decoder) uint32() uint32 {
	x := d.order.Uint32(d.buf[0:4]);
	d.buf = d.buf[4:len(d.buf)];
	return x;
}

func (d *decoder) uint64() uint64 {
	x := d.order.Uint64(d.buf[0:8]);
	d.buf = d.buf[8:len(d.buf)];
	return x;
}

func (d *decoder) int8() int8 {
	return int8(d.uint8());
}

func (d *decoder) int16() int16 {
	return int16(d.uint16());
}

func (d *decoder) int32() int32 {
	return int32(d.uint32());
}

func (d *decoder) int64() int64 {
	return int64(d.uint64());
}

func (d *decoder) value(v reflect.Value) {
	switch v := v.(type) {
	case *reflect.ArrayValue:
		l := v.Len();
		for i := 0; i < l; i++ {
			d.value(v.Elem(i));
		}
	case *reflect.StructValue:
		l := v.NumField();
		for i := 0; i < l; i++ {
			d.value(v.Field(i));
		}

	case *reflect.Uint8Value:
		v.Set(d.uint8());
	case *reflect.Uint16Value:
		v.Set(d.uint16());
	case *reflect.Uint32Value:
		v.Set(d.uint32());
	case *reflect.Uint64Value:
		v.Set(d.uint64());
	case *reflect.Int8Value:
		v.Set(d.int8());
	case *reflect.Int16Value:
		v.Set(d.int16());
	case *reflect.Int32Value:
		v.Set(d.int32());
	case *reflect.Int64Value:
		v.Set(d.int64());
	case *reflect.Float32Value:
		v.Set(math.Float32frombits(d.uint32()));
	case *reflect.Float64Value:
		v.Set(math.Float64frombits(d.uint64()));
	}
}
