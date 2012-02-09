// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package binary implements translation between
// unsigned integer values and byte sequences
// and the reading and writing of fixed-size values.
// A fixed-size value is either a fixed-size arithmetic
// type (int8, uint8, int16, float32, complex64, ...)
// or an array or struct containing only fixed-size values.
package binary

import (
	"errors"
	"io"
	"math"
	"reflect"
)

// A ByteOrder specifies how to convert byte sequences into
// 16-, 32-, or 64-bit unsigned integers.
type ByteOrder interface {
	Uint16([]byte) uint16
	Uint32([]byte) uint32
	Uint64([]byte) uint64
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
// Bytes read from r are decoded using the specified byte order
// and written to successive fields of the data.
func Read(r io.Reader, order ByteOrder, data interface{}) error {
	// Fast path for basic types.
	if n := intDestSize(data); n != 0 {
		var b [8]byte
		bs := b[:n]
		if _, err := io.ReadFull(r, bs); err != nil {
			return err
		}
		switch v := data.(type) {
		case *int8:
			*v = int8(b[0])
		case *uint8:
			*v = b[0]
		case *int16:
			*v = int16(order.Uint16(bs))
		case *uint16:
			*v = order.Uint16(bs)
		case *int32:
			*v = int32(order.Uint32(bs))
		case *uint32:
			*v = order.Uint32(bs)
		case *int64:
			*v = int64(order.Uint64(bs))
		case *uint64:
			*v = order.Uint64(bs)
		}
		return nil
	}

	// Fallback to reflect-based.
	var v reflect.Value
	switch d := reflect.ValueOf(data); d.Kind() {
	case reflect.Ptr:
		v = d.Elem()
	case reflect.Slice:
		v = d
	default:
		return errors.New("binary.Read: invalid type " + d.Type().String())
	}
	size := dataSize(v)
	if size < 0 {
		return errors.New("binary.Read: invalid type " + v.Type().String())
	}
	d := &decoder{order: order, buf: make([]byte, size)}
	if _, err := io.ReadFull(r, d.buf); err != nil {
		return err
	}
	d.value(v)
	return nil
}

// Write writes the binary representation of data into w.
// Data must be a fixed-size value or a slice of fixed-size
// values, or a pointer to such data.
// Bytes written to w are encoded using the specified byte order
// and read from successive fields of the data.
func Write(w io.Writer, order ByteOrder, data interface{}) error {
	// Fast path for basic types.
	var b [8]byte
	var bs []byte
	switch v := data.(type) {
	case *int8:
		bs = b[:1]
		b[0] = byte(*v)
	case int8:
		bs = b[:1]
		b[0] = byte(v)
	case *uint8:
		bs = b[:1]
		b[0] = *v
	case uint8:
		bs = b[:1]
		b[0] = byte(v)
	case *int16:
		bs = b[:2]
		order.PutUint16(bs, uint16(*v))
	case int16:
		bs = b[:2]
		order.PutUint16(bs, uint16(v))
	case *uint16:
		bs = b[:2]
		order.PutUint16(bs, *v)
	case uint16:
		bs = b[:2]
		order.PutUint16(bs, v)
	case *int32:
		bs = b[:4]
		order.PutUint32(bs, uint32(*v))
	case int32:
		bs = b[:4]
		order.PutUint32(bs, uint32(v))
	case *uint32:
		bs = b[:4]
		order.PutUint32(bs, *v)
	case uint32:
		bs = b[:4]
		order.PutUint32(bs, v)
	case *int64:
		bs = b[:8]
		order.PutUint64(bs, uint64(*v))
	case int64:
		bs = b[:8]
		order.PutUint64(bs, uint64(v))
	case *uint64:
		bs = b[:8]
		order.PutUint64(bs, *v)
	case uint64:
		bs = b[:8]
		order.PutUint64(bs, v)
	}
	if bs != nil {
		_, err := w.Write(bs)
		return err
	}
	v := reflect.Indirect(reflect.ValueOf(data))
	size := dataSize(v)
	if size < 0 {
		return errors.New("binary.Write: invalid type " + v.Type().String())
	}
	buf := make([]byte, size)
	e := &encoder{order: order, buf: buf}
	e.value(v)
	_, err := w.Write(buf)
	return err
}

// Size returns how many bytes Write would generate to encode the value v, which
// must be a fixed-size value or a slice of fixed-size values, or a pointer to such data.
func Size(v interface{}) int {
	return dataSize(reflect.Indirect(reflect.ValueOf(v)))
}

// dataSize returns the number of bytes the actual data represented by v occupies in memory.
// For compound structures, it sums the sizes of the elements. Thus, for instance, for a slice
// it returns the length of the slice times the element size and does not count the memory
// occupied by the header.
func dataSize(v reflect.Value) int {
	if v.Kind() == reflect.Slice {
		elem := sizeof(v.Type().Elem())
		if elem < 0 {
			return -1
		}
		return v.Len() * elem
	}
	return sizeof(v.Type())
}

func sizeof(t reflect.Type) int {
	switch t.Kind() {
	case reflect.Array:
		n := sizeof(t.Elem())
		if n < 0 {
			return -1
		}
		return t.Len() * n

	case reflect.Struct:
		sum := 0
		for i, n := 0, t.NumField(); i < n; i++ {
			s := sizeof(t.Field(i).Type)
			if s < 0 {
				return -1
			}
			sum += s
		}
		return sum

	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		return int(t.Size())
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
	switch v.Kind() {
	case reflect.Array:
		l := v.Len()
		for i := 0; i < l; i++ {
			d.value(v.Index(i))
		}

	case reflect.Struct:
		l := v.NumField()
		for i := 0; i < l; i++ {
			d.value(v.Field(i))
		}

	case reflect.Slice:
		l := v.Len()
		for i := 0; i < l; i++ {
			d.value(v.Index(i))
		}

	case reflect.Int8:
		v.SetInt(int64(d.int8()))
	case reflect.Int16:
		v.SetInt(int64(d.int16()))
	case reflect.Int32:
		v.SetInt(int64(d.int32()))
	case reflect.Int64:
		v.SetInt(d.int64())

	case reflect.Uint8:
		v.SetUint(uint64(d.uint8()))
	case reflect.Uint16:
		v.SetUint(uint64(d.uint16()))
	case reflect.Uint32:
		v.SetUint(uint64(d.uint32()))
	case reflect.Uint64:
		v.SetUint(d.uint64())

	case reflect.Float32:
		v.SetFloat(float64(math.Float32frombits(d.uint32())))
	case reflect.Float64:
		v.SetFloat(math.Float64frombits(d.uint64()))

	case reflect.Complex64:
		v.SetComplex(complex(
			float64(math.Float32frombits(d.uint32())),
			float64(math.Float32frombits(d.uint32())),
		))
	case reflect.Complex128:
		v.SetComplex(complex(
			math.Float64frombits(d.uint64()),
			math.Float64frombits(d.uint64()),
		))
	}
}

func (e *encoder) value(v reflect.Value) {
	switch v.Kind() {
	case reflect.Array:
		l := v.Len()
		for i := 0; i < l; i++ {
			e.value(v.Index(i))
		}

	case reflect.Struct:
		l := v.NumField()
		for i := 0; i < l; i++ {
			e.value(v.Field(i))
		}

	case reflect.Slice:
		l := v.Len()
		for i := 0; i < l; i++ {
			e.value(v.Index(i))
		}

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch v.Type().Kind() {
		case reflect.Int8:
			e.int8(int8(v.Int()))
		case reflect.Int16:
			e.int16(int16(v.Int()))
		case reflect.Int32:
			e.int32(int32(v.Int()))
		case reflect.Int64:
			e.int64(v.Int())
		}

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		switch v.Type().Kind() {
		case reflect.Uint8:
			e.uint8(uint8(v.Uint()))
		case reflect.Uint16:
			e.uint16(uint16(v.Uint()))
		case reflect.Uint32:
			e.uint32(uint32(v.Uint()))
		case reflect.Uint64:
			e.uint64(v.Uint())
		}

	case reflect.Float32, reflect.Float64:
		switch v.Type().Kind() {
		case reflect.Float32:
			e.uint32(math.Float32bits(float32(v.Float())))
		case reflect.Float64:
			e.uint64(math.Float64bits(v.Float()))
		}

	case reflect.Complex64, reflect.Complex128:
		switch v.Type().Kind() {
		case reflect.Complex64:
			x := v.Complex()
			e.uint32(math.Float32bits(float32(real(x))))
			e.uint32(math.Float32bits(float32(imag(x))))
		case reflect.Complex128:
			x := v.Complex()
			e.uint64(math.Float64bits(real(x)))
			e.uint64(math.Float64bits(imag(x)))
		}
	}
}

// intDestSize returns the size of the integer that ptrType points to,
// or 0 if the type is not supported.
func intDestSize(ptrType interface{}) int {
	switch ptrType.(type) {
	case *int8, *uint8:
		return 1
	case *int16, *uint16:
		return 2
	case *int32, *uint32:
		return 4
	case *int64, *uint64:
		return 8
	}
	return 0
}
