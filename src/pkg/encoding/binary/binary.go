// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements translation between
// unsigned integer values and byte sequences.
package binary

import (
	"math"
	"io"
	"os"
	"reflect"
)

// A ByteOrder specifies how to convert byte sequences into
// 16-, 32-, or 64-bit unsigned integers.
type ByteOrder interface {
	Uint16(b []byte) uint16
	Uint32(b []byte) uint32
	Uint64(b []byte) uint64
	PutUint16([]byte, uint16)
	PutUint32([]byte, uint32)
	PutUint64([]byte, uint64)
	String() string
}

// This is byte instead of struct{} so that it can be compared,
// allowing, e.g., order == binary.LittleEndian.
type unused byte

// LittleEndian is the little-endian implementation of ByteOrder.
var LittleEndian littleEndian

// BigEndian is the big-endian implementation of ByteOrder.
var BigEndian bigEndian

type littleEndian unused

func (littleEndian) Uint16(b []byte) uint16 { return uint16(b[0]) | uint16(b[1])<<8 }

func (littleEndian) PutUint16(b []byte, v uint16) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
}

func (littleEndian) Uint32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func (littleEndian) PutUint32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

func (littleEndian) Uint64(b []byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func (littleEndian) PutUint64(b []byte, v uint64) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}

func (littleEndian) String() string { return "LittleEndian" }

func (littleEndian) GoString() string { return "binary.LittleEndian" }

type bigEndian unused

func (bigEndian) Uint16(b []byte) uint16 { return uint16(b[1]) | uint16(b[0])<<8 }

func (bigEndian) PutUint16(b []byte, v uint16) {
	b[0] = byte(v >> 8)
	b[1] = byte(v)
}

func (bigEndian) Uint32(b []byte) uint32 {
	return uint32(b[3]) | uint32(b[2])<<8 | uint32(b[1])<<16 | uint32(b[0])<<24
}

func (bigEndian) PutUint32(b []byte, v uint32) {
	b[0] = byte(v >> 24)
	b[1] = byte(v >> 16)
	b[2] = byte(v >> 8)
	b[3] = byte(v)
}

func (bigEndian) Uint64(b []byte) uint64 {
	return uint64(b[7]) | uint64(b[6])<<8 | uint64(b[5])<<16 | uint64(b[4])<<24 |
		uint64(b[3])<<32 | uint64(b[2])<<40 | uint64(b[1])<<48 | uint64(b[0])<<56
}

func (bigEndian) PutUint64(b []byte, v uint64) {
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)
}

func (bigEndian) String() string { return "BigEndian" }

func (bigEndian) GoString() string { return "binary.BigEndian" }

// Read reads structured binary data from r into data.
// Data must be a pointer to a fixed-size value or a slice
// of fixed-size values.
// A fixed-size value is either a fixed-size arithmetic
// type (int8, uint8, int16, float32, complex64, ...)
// or an array or struct containing only fixed-size values.
// Bytes read from r are decoded using the specified byte order
// and written to successive fields of the data.
func Read(r io.Reader, order ByteOrder, data interface{}) os.Error {
	var v reflect.Value
	switch d := reflect.NewValue(data).(type) {
	case *reflect.PtrValue:
		v = d.Elem()
	case *reflect.SliceValue:
		v = d
	default:
		return os.NewError("binary.Read: invalid type " + d.Type().String())
	}
	size := TotalSize(v)
	if size < 0 {
		return os.NewError("binary.Read: invalid type " + v.Type().String())
	}
	d := &decoder{order: order, buf: make([]byte, size)}
	if _, err := io.ReadFull(r, d.buf); err != nil {
		return err
	}
	d.value(v)
	return nil
}

// Write writes the binary representation of data into w.
// Data must be a fixed-size value or a pointer to
// a fixed-size value.
// A fixed-size value is either a fixed-size arithmetic
// type (int8, uint8, int16, float32, complex64, ...)
// or an array or struct containing only fixed-size values.
// Bytes written to w are encoded using the specified byte order
// and read from successive fields of the data.
func Write(w io.Writer, order ByteOrder, data interface{}) os.Error {
	v := reflect.Indirect(reflect.NewValue(data))
	size := TotalSize(v)
	if size < 0 {
		return os.NewError("binary.Write: invalid type " + v.Type().String())
	}
	buf := make([]byte, size)
	e := &encoder{order: order, buf: buf}
	e.value(v)
	_, err := w.Write(buf)
	return err
}

func TotalSize(v reflect.Value) int {
	if sv, ok := v.(*reflect.SliceValue); ok {
		elem := sizeof(v.Type().(*reflect.SliceType).Elem())
		if elem < 0 {
			return -1
		}
		return sv.Len() * elem
	}
	return sizeof(v.Type())
}

func sizeof(v reflect.Type) int {
	switch t := v.(type) {
	case *reflect.ArrayType:
		n := sizeof(t.Elem())
		if n < 0 {
			return -1
		}
		return t.Len() * n

	case *reflect.StructType:
		sum := 0
		for i, n := 0, t.NumField(); i < n; i++ {
			s := sizeof(t.Field(i).Type)
			if s < 0 {
				return -1
			}
			sum += s
		}
		return sum

	case *reflect.UintType, *reflect.IntType, *reflect.FloatType, *reflect.ComplexType:
		switch t := t.Kind(); t {
		case reflect.Int, reflect.Uint, reflect.Uintptr:
			return -1
		}
		return int(v.Size())
	}
	return -1
}

type decoder struct {
	order ByteOrder
	buf   []byte
}

type encoder struct {
	order ByteOrder
	buf   []byte
}

func (d *decoder) uint8() uint8 {
	x := d.buf[0]
	d.buf = d.buf[1:]
	return x
}

func (e *encoder) uint8(x uint8) {
	e.buf[0] = x
	e.buf = e.buf[1:]
}

func (d *decoder) uint16() uint16 {
	x := d.order.Uint16(d.buf[0:2])
	d.buf = d.buf[2:]
	return x
}

func (e *encoder) uint16(x uint16) {
	e.order.PutUint16(e.buf[0:2], x)
	e.buf = e.buf[2:]
}

func (d *decoder) uint32() uint32 {
	x := d.order.Uint32(d.buf[0:4])
	d.buf = d.buf[4:]
	return x
}

func (e *encoder) uint32(x uint32) {
	e.order.PutUint32(e.buf[0:4], x)
	e.buf = e.buf[4:]
}

func (d *decoder) uint64() uint64 {
	x := d.order.Uint64(d.buf[0:8])
	d.buf = d.buf[8:]
	return x
}

func (e *encoder) uint64(x uint64) {
	e.order.PutUint64(e.buf[0:8], x)
	e.buf = e.buf[8:]
}

func (d *decoder) int8() int8 { return int8(d.uint8()) }

func (e *encoder) int8(x int8) { e.uint8(uint8(x)) }

func (d *decoder) int16() int16 { return int16(d.uint16()) }

func (e *encoder) int16(x int16) { e.uint16(uint16(x)) }

func (d *decoder) int32() int32 { return int32(d.uint32()) }

func (e *encoder) int32(x int32) { e.uint32(uint32(x)) }

func (d *decoder) int64() int64 { return int64(d.uint64()) }

func (e *encoder) int64(x int64) { e.uint64(uint64(x)) }

func (d *decoder) value(v reflect.Value) {
	switch v := v.(type) {
	case *reflect.ArrayValue:
		l := v.Len()
		for i := 0; i < l; i++ {
			d.value(v.Elem(i))
		}
	case *reflect.StructValue:
		l := v.NumField()
		for i := 0; i < l; i++ {
			d.value(v.Field(i))
		}

	case *reflect.SliceValue:
		l := v.Len()
		for i := 0; i < l; i++ {
			d.value(v.Elem(i))
		}

	case *reflect.IntValue:
		switch v.Type().Kind() {
		case reflect.Int8:
			v.Set(int64(d.int8()))
		case reflect.Int16:
			v.Set(int64(d.int16()))
		case reflect.Int32:
			v.Set(int64(d.int32()))
		case reflect.Int64:
			v.Set(d.int64())
		}

	case *reflect.UintValue:
		switch v.Type().Kind() {
		case reflect.Uint8:
			v.Set(uint64(d.uint8()))
		case reflect.Uint16:
			v.Set(uint64(d.uint16()))
		case reflect.Uint32:
			v.Set(uint64(d.uint32()))
		case reflect.Uint64:
			v.Set(d.uint64())
		}

	case *reflect.FloatValue:
		switch v.Type().Kind() {
		case reflect.Float32:
			v.Set(float64(math.Float32frombits(d.uint32())))
		case reflect.Float64:
			v.Set(math.Float64frombits(d.uint64()))
		}

	case *reflect.ComplexValue:
		switch v.Type().Kind() {
		case reflect.Complex64:
			v.Set(complex(
				float64(math.Float32frombits(d.uint32())),
				float64(math.Float32frombits(d.uint32())),
			))
		case reflect.Complex128:
			v.Set(complex(
				math.Float64frombits(d.uint64()),
				math.Float64frombits(d.uint64()),
			))
		}
	}
}

func (e *encoder) value(v reflect.Value) {
	switch v := v.(type) {
	case *reflect.ArrayValue:
		l := v.Len()
		for i := 0; i < l; i++ {
			e.value(v.Elem(i))
		}
	case *reflect.StructValue:
		l := v.NumField()
		for i := 0; i < l; i++ {
			e.value(v.Field(i))
		}
	case *reflect.SliceValue:
		l := v.Len()
		for i := 0; i < l; i++ {
			e.value(v.Elem(i))
		}

	case *reflect.IntValue:
		switch v.Type().Kind() {
		case reflect.Int8:
			e.int8(int8(v.Get()))
		case reflect.Int16:
			e.int16(int16(v.Get()))
		case reflect.Int32:
			e.int32(int32(v.Get()))
		case reflect.Int64:
			e.int64(v.Get())
		}

	case *reflect.UintValue:
		switch v.Type().Kind() {
		case reflect.Uint8:
			e.uint8(uint8(v.Get()))
		case reflect.Uint16:
			e.uint16(uint16(v.Get()))
		case reflect.Uint32:
			e.uint32(uint32(v.Get()))
		case reflect.Uint64:
			e.uint64(v.Get())
		}

	case *reflect.FloatValue:
		switch v.Type().Kind() {
		case reflect.Float32:
			e.uint32(math.Float32bits(float32(v.Get())))
		case reflect.Float64:
			e.uint64(math.Float64bits(v.Get()))
		}

	case *reflect.ComplexValue:
		switch v.Type().Kind() {
		case reflect.Complex64:
			x := v.Get()
			e.uint32(math.Float32bits(float32(real(x))))
			e.uint32(math.Float32bits(float32(imag(x))))
		case reflect.Complex128:
			x := v.Get()
			e.uint64(math.Float64bits(real(x)))
			e.uint64(math.Float64bits(imag(x)))
		}
	}
}
