// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
)

// byteReader tests
func TestByteReaderRemaining(t *T) {
	tests := []struct {
		name          string
		data          []byte
		readSize      int
		wantRemaining int
	}{
		{"empty", []byte{}, 0, 0},
		{"full", []byte{1, 2, 3, 4}, 0, 4},
		{"after read", []byte{1, 2, 3, 4}, 2, 2},
		{"exhausted", []byte{1, 2}, 2, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *T) {
			r := newByteReader(tt.data)
			if tt.readSize > 0 {
				r.readBytes(tt.readSize)
			}
			if got := r.remaining(); got != tt.wantRemaining {
				t.Errorf("remaining() = %d, want %d",
					got, tt.wantRemaining)
			}
		})
	}
}

func TestByteReaderReadBytes(t *T) {
	t.Run("exact", func(t *T) {
		r := newByteReader([]byte{1, 2, 3})
		got := r.readBytes(3)
		if !bytes.Equal(got, []byte{1, 2, 3}) {
			t.Errorf("readBytes(3) = %v, want %v",
				got, []byte{1, 2, 3})
		}
	})

	t.Run("partial", func(t *T) {
		r := newByteReader([]byte{1, 2, 3, 4})
		got := r.readBytes(2)
		if !bytes.Equal(got, []byte{1, 2}) {
			t.Errorf("readBytes(2) = %v, want %v",
				got, []byte{1, 2})
		}
		if r.remaining() != 2 {
			t.Errorf("remaining() = %d, want 2",
				r.remaining())
		}
	})

	t.Run("overflow returns available", func(t *T) {
		r := newByteReader([]byte{1, 2})
		got := r.readBytes(5)
		if !bytes.Equal(got, []byte{1, 2}) {
			t.Errorf("readBytes(5) = %v, want %v",
				got, []byte{1, 2})
		}
	})

	t.Run("empty source", func(t *T) {
		r := newByteReader([]byte{})
		got := r.readBytes(3)
		if got != nil {
			t.Errorf("readBytes(3) from empty = %v, want nil",
				got)
		}
	})

	t.Run("zero size", func(t *T) {
		r := newByteReader([]byte{1, 2, 3})
		got := r.readBytes(0)
		if got != nil {
			t.Errorf("readBytes(0) = %v, want nil",
				got)
		}
	})

	t.Run("negative size", func(t *T) {
		r := newByteReader([]byte{1, 2, 3})
		got := r.readBytes(-1)
		if got != nil {
			t.Errorf("readBytes(-1) = %v, want nil",
				got)
		}
	})
}

func TestByteReaderReadBytesPadded(t *T) {
	t.Run("exact", func(t *T) {
		r := newByteReader([]byte{1, 2, 3})
		got := r.readBytesPadded(3)
		if !bytes.Equal(got, []byte{1, 2, 3}) {
			t.Errorf("readBytesPadded(3) = %v, want %v",
				got, []byte{1, 2, 3})
		}
	})

	t.Run("overflow with padding", func(t *T) {
		r := newByteReader([]byte{1, 2})
		got := r.readBytesPadded(5)
		if !bytes.Equal(got, []byte{1, 2, 0, 0, 0}) {
			t.Errorf("readBytesPadded(5) = %v, want %v",
				got, []byte{1, 2, 0, 0, 0})
		}
	})

	t.Run("empty source", func(t *T) {
		r := newByteReader([]byte{})
		got := r.readBytesPadded(3)
		if !bytes.Equal(got, []byte{0, 0, 0}) {
			t.Errorf("readBytesPadded(3) from empty = %v, want %v",
				got, []byte{0, 0, 0})
		}
	})
}

func TestByteReaderReadUint32(t *T) {
	t.Run("normal", func(t *T) {
		r := newByteReader([]byte{0x00, 0x00, 0x01, 0x00})
		got := r.readUint32()
		if got != 256 {
			t.Errorf("readUint32() = %d, want 256", got)
		}
	})

	t.Run("max value", func(t *T) {
		r := newByteReader([]byte{0xFF, 0xFF, 0xFF, 0xFF})
		got := r.readUint32()
		if got != 0xFFFFFFFF {
			t.Errorf("readUint32() = %d, want %d",
				got, uint32(0xFFFFFFFF))
		}
	})

	t.Run("short input padded", func(t *T) {
		r := newByteReader([]byte{0x01, 0x02})
		got := r.readUint32()
		// 0x01, 0x02, 0x00, 0x00 in big-endian = 0x01020000
		if got != 0x01020000 {
			t.Errorf("readUint32() from short = %x, want %x",
				got, 0x01020000)
		}
	})
}

func TestByteReaderReadUint64(t *T) {
	t.Run("normal", func(t *T) {
		r := newByteReader([]byte{0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x01, 0x00},
		)
		got := r.readUint64()
		if got != 256 {
			t.Errorf("readUint64() = %d, want 256",
				got)
		}
	})

	t.Run("large value", func(t *T) {
		r := newByteReader([]byte{0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00, 0x00},
		)
		got := r.readUint64()
		if got != 0x0000000100000000 {
			t.Errorf("readUint64() = %x, want %x",
				got, uint64(0x0000000100000000))
		}
	})
}

func TestByteReaderReadFixedValue(t *T) {
	tests := []struct {
		name string
		data []byte
		typ  reflect.Type
		want any
	}{
		{"bool true", []byte{0x01}, reflect.TypeOf(false), true},
		{"bool false", []byte{0x00}, reflect.TypeOf(false), false},
		{"int8 positive", []byte{0x7F}, reflect.TypeOf(int8(0)), int8(127)},
		{"int8 negative", []byte{0xFF}, reflect.TypeOf(int8(0)), int8(-1)},
		{"int16", []byte{0x01, 0x02}, reflect.TypeOf(int16(0)), int16(258)},
		{"int16 negative", []byte{0xFF, 0xFE}, reflect.TypeOf(int16(0)), int16(-2)},
		{"int32", []byte{0x00, 0x00, 0x01, 0x00}, reflect.TypeOf(int32(0)), int32(256)},
		{"int32 negative", []byte{0xFF, 0xFF, 0xFF, 0xFF}, reflect.TypeOf(int32(0)), int32(-1)},
		{"int64", []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00},
			reflect.TypeOf(int64(0)), int64(256)},
		{"uint8", []byte{0xFF}, reflect.TypeOf(uint8(0)), uint8(255)},
		{"uint16", []byte{0x01, 0x02}, reflect.TypeOf(uint16(0)), uint16(258)},
		{"uint32", []byte{0x00, 0x00, 0x01, 0x00}, reflect.TypeOf(uint32(0)), uint32(256)},
		{"uint64", []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00},
			reflect.TypeOf(uint64(0)), uint64(256)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *T) {
			r := newByteReader(tt.data)
			got := r.readFixedValue(tt.typ)
			if got != tt.want {
				t.Errorf("readFixedValue(%v) = %v, want %v",
					tt.typ, got, tt.want)
			}
		})
	}

	// Float tests need special handling
	t.Run("float32", func(t *T) {
		var buf [4]byte
		binary.BigEndian.PutUint32(buf[:], math.Float32bits(3.14))
		r := newByteReader(buf[:])
		got := r.readFixedValue(reflect.TypeOf(float32(0)))
		if got != float32(3.14) {
			t.Errorf("readFixedValue(float32) = %v, want 3.14", got)
		}
	})

	t.Run("float64", func(t *T) {
		var buf [8]byte
		binary.BigEndian.PutUint64(buf[:], math.Float64bits(3.14159))
		r := newByteReader(buf[:])
		got := r.readFixedValue(reflect.TypeOf(float64(0)))
		if got != float64(3.14159) {
			t.Errorf("readFixedValue(float64) = %v, want 3.14159", got)
		}
	})
}

func TestByteReaderSequentialReads(t *T) {
	data := []byte{
		0x01,       // int8
		0x00, 0x02, // int16
		0x00, 0x00, 0x00, 0x03, // int32
	}
	r := newByteReader(data)

	v1 := r.readFixedValue(reflect.TypeOf(int8(0)))
	if v1 != int8(1) {
		t.Errorf("first read = %v, want 1", v1)
	}

	v2 := r.readFixedValue(reflect.TypeOf(int16(0)))
	if v2 != int16(2) {
		t.Errorf("second read = %v, want 2", v2)
	}

	v3 := r.readFixedValue(reflect.TypeOf(int32(0)))
	if v3 != int32(3) {
		t.Errorf("third read = %v, want 3", v3)
	}

	if r.remaining() != 0 {
		t.Errorf("remaining() = %d, want 0", r.remaining())
	}
}

// fixedTypeSize tests
func TestFixedTypeSize(t *T) {
	tests := []struct {
		typ  reflect.Type
		want int
	}{
		{reflect.TypeOf(true), 1},
		{reflect.TypeOf(int8(0)), 1},
		{reflect.TypeOf(uint8(0)), 1},
		{reflect.TypeOf(int16(0)), 2},
		{reflect.TypeOf(uint16(0)), 2},
		{reflect.TypeOf(int32(0)), 4},
		{reflect.TypeOf(uint32(0)), 4},
		{reflect.TypeOf(float32(0)), 4},
		{reflect.TypeOf(int64(0)), 8},
		{reflect.TypeOf(uint64(0)), 8},
		{reflect.TypeOf(int(0)), 8},
		{reflect.TypeOf(uint(0)), 8},
		{reflect.TypeOf(float64(0)), 8},
		{reflect.TypeOf(""), 0},       // dynamic
		{reflect.TypeOf([]byte{}), 0}, // dynamic
	}

	for _, tt := range tests {
		t.Run(tt.typ.String(), func(t *T) {
			if got := fixedTypeSize(tt.typ); got != tt.want {
				t.Errorf("fixedTypeSize(%v) = %d, want %d",
					tt.typ, got, tt.want)
			}
		})
	}
}

// Serialization tests
func TestSerializeLibfuzzerBytes(t *T) {
	t.Run("single int32", func(t *T) {
		data := serializeLibfuzzerBytes([]any{int32(256)})
		want := []byte{0x00, 0x00, 0x01, 0x00}
		if !bytes.Equal(data, want) {
			t.Errorf("got %v, want %v", data, want)
		}
	})

	t.Run("single string", func(t *T) {
		data := serializeLibfuzzerBytes([]any{"hello"})
		want := []byte("hello")
		if !bytes.Equal(data, want) {
			t.Errorf("got %v, want %v", data, want)
		}
	})

	t.Run("single []byte", func(t *T) {
		data := serializeLibfuzzerBytes([]any{[]byte{1, 2, 3}})
		want := []byte{1, 2, 3}
		if !bytes.Equal(data, want) {
			t.Errorf("got %v, want %v", data, want)
		}
	})

	t.Run("two strings with weights", func(t *T) {
		data := serializeLibfuzzerBytes([]any{"hello", "world"})
		// Should be: [uint32(5)][uint32(5)][helloworld]
		var expected bytes.Buffer
		binary.Write(&expected, binary.BigEndian, uint32(5))
		binary.Write(&expected, binary.BigEndian, uint32(5))
		expected.WriteString("helloworld")
		if !bytes.Equal(data, expected.Bytes()) {
			t.Errorf("got %v, want %v", data, expected.Bytes())
		}
	})

	t.Run("int32 and string", func(t *T) {
		data := serializeLibfuzzerBytes([]any{int32(42), "test"})
		var expected bytes.Buffer
		binary.Write(&expected, binary.BigEndian, int32(42))
		expected.WriteString("test")
		if !bytes.Equal(data, expected.Bytes()) {
			t.Errorf("got %v, want %v", data, expected.Bytes())
		}
	})

	t.Run("all fixed types", func(t *T) {
		vals := []any{
			true,
			int8(-1),
			int16(1000),
			int32(100000),
			int64(10000000000),
			uint8(255),
			uint16(65535),
			uint32(4000000000),
			uint64(10000000000000000000),
			float32(3.14),
			float64(2.71828),
		}
		data := serializeLibfuzzerBytes(vals)

		// Verify length matches expected sizes
		expectedLen := 1 + 1 + 2 + 4 + 8 + 1 + 2 + 4 + 8 + 4 + 8 // 43 bytes
		if len(data) != expectedLen {
			t.Errorf("len = %d, want %d", len(data), expectedLen)
		}
	})
}

// Deserialization tests
func TestDeserializeLibfuzzerBytes(t *T) {
	t.Run("single []byte", func(t *T) {
		data := []byte{1, 2, 3, 4, 5}
		types := []reflect.Type{reflect.TypeOf([]byte{})}
		vals := deserializeLibfuzzerBytes(data, types)
		if len(vals) != 1 {
			t.Fatalf("got %d vals, want 1", len(vals))
		}
		if !bytes.Equal(vals[0].([]byte), data) {
			t.Errorf("vals[0] = %v, want %v", vals[0], data)
		}
	})

	t.Run("single string", func(t *T) {
		data := []byte("hello world")
		types := []reflect.Type{reflect.TypeOf("")}
		vals := deserializeLibfuzzerBytes(data, types)
		if len(vals) != 1 {
			t.Fatalf("got %d vals, want 1", len(vals))
		}
		if vals[0] != "hello world" {
			t.Errorf("vals[0] = %v, want 'hello world'", vals[0])
		}
	})

	t.Run("single int32", func(t *T) {
		data := []byte{0x00, 0x00, 0x01, 0x00}
		types := []reflect.Type{reflect.TypeOf(int32(0))}
		vals := deserializeLibfuzzerBytes(data, types)
		if len(vals) != 1 {
			t.Fatalf("got %d vals, want 1", len(vals))
		}
		if vals[0] != int32(256) {
			t.Errorf("vals[0] = %v, want 256", vals[0])
		}
	})

	t.Run("int32 and string", func(t *T) {
		data := []byte{0x00, 0x00, 0x00, 0x2A, 'h', 'i'}
		types := []reflect.Type{
			reflect.TypeOf(int32(0)),
			reflect.TypeOf(""),
		}
		vals := deserializeLibfuzzerBytes(data, types)
		if len(vals) != 2 {
			t.Fatalf("got %d vals, want 2", len(vals))
		}
		if vals[0] != int32(42) {
			t.Errorf("vals[0] = %v, want 42", vals[0])
		}
		if vals[1] != "hi" {
			t.Errorf("vals[1] = %v, want 'hi'", vals[1])
		}
	})

	t.Run("two strings with weights", func(t *T) {
		var data bytes.Buffer
		binary.Write(&data, binary.BigEndian, uint32(5))
		binary.Write(&data, binary.BigEndian, uint32(5))
		data.WriteString("helloworld")

		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}
		vals := deserializeLibfuzzerBytes(data.Bytes(), types)
		if len(vals) != 2 {
			t.Fatalf("got %d vals, want 2", len(vals))
		}
		if vals[0] != "hello" {
			t.Errorf("vals[0] = %v, want 'hello'", vals[0])
		}
		if vals[1] != "world" {
			t.Errorf("vals[1] = %v, want 'world'", vals[1])
		}
	})

	t.Run("three strings with weights", func(t *T) {
		var data bytes.Buffer
		binary.Write(&data, binary.BigEndian, uint32(3))
		binary.Write(&data, binary.BigEndian, uint32(4))
		binary.Write(&data, binary.BigEndian, uint32(5))
		data.WriteString("abcdefghijkl")

		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}
		vals := deserializeLibfuzzerBytes(data.Bytes(), types)
		if len(vals) != 3 {
			t.Fatalf("got %d vals, want 3", len(vals))
		}
		if vals[0] != "abc" {
			t.Errorf("vals[0] = %v, want 'abc'", vals[0])
		}
		if vals[1] != "defg" {
			t.Errorf("vals[1] = %v, want 'defg'", vals[1])
		}
		if vals[2] != "hijkl" {
			t.Errorf("vals[2] = %v, want 'hijkl'", vals[2])
		}
	})

	t.Run("empty input for string", func(t *T) {
		vals := deserializeLibfuzzerBytes(
			[]byte{},
			[]reflect.Type{reflect.TypeOf("")},
		)
		if len(vals) != 1 {
			t.Fatalf("got %d vals, want 1", len(vals))
		}
		if vals[0] != "" {
			t.Errorf("vals[0] = %v, want empty string", vals[0])
		}
	})

	t.Run("multiple fixed types", func(t *T) {
		var data bytes.Buffer
		data.WriteByte(1)                                    // bool = true
		data.WriteByte(0xFF)                                 // int8 = -1
		binary.Write(&data, binary.BigEndian, uint16(1000))  // uint16 = 1000
		binary.Write(&data, binary.BigEndian, int32(-12345)) // int32 = -12345

		types := []reflect.Type{
			reflect.TypeOf(true),
			reflect.TypeOf(int8(0)),
			reflect.TypeOf(uint16(0)),
			reflect.TypeOf(int32(0)),
		}
		vals := deserializeLibfuzzerBytes(data.Bytes(), types)
		if len(vals) != 4 {
			t.Fatalf("got %d vals, want 4", len(vals))
		}
		if vals[0] != true {
			t.Errorf("vals[0] = %v, want true", vals[0])
		}
		if vals[1] != int8(-1) {
			t.Errorf("vals[1] = %v, want -1", vals[1])
		}
		if vals[2] != uint16(1000) {
			t.Errorf("vals[2] = %v, want 1000", vals[2])
		}
		if vals[3] != int32(-12345) {
			t.Errorf("vals[3] = %v, want -12345", vals[3])
		}
	})

	t.Run("empty types", func(t *T) {
		vals := deserializeLibfuzzerBytes([]byte{1, 2, 3}, nil)
		if vals != nil {
			t.Errorf("expected nil for empty types, got %v", vals)
		}
	})
}

// Round-trip tests (serialize -> deserialize)
func TestRoundTrip(t *T) {
	t.Run("single string", func(t *T) {
		original := []any{"hello world"}
		types := []reflect.Type{reflect.TypeOf("")}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if vals[0] != "hello world" {
			t.Errorf("got %q, want %q", vals[0], "hello world")
		}
	})

	t.Run("single []byte", func(t *T) {
		original := []any{[]byte{0, 1, 2, 255, 254}}
		types := []reflect.Type{reflect.TypeOf([]byte{})}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if !bytes.Equal(vals[0].([]byte), []byte{0, 1, 2, 255, 254}) {
			t.Errorf("got %v, want %v", vals[0], []byte{0, 1, 2, 255, 254})
		}
	})

	t.Run("two strings", func(t *T) {
		original := []any{"hello", "world"}
		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if vals[0] != "hello" {
			t.Errorf("vals[0] = %q, want %q", vals[0], "hello")
		}
		if vals[1] != "world" {
			t.Errorf("vals[1] = %q, want %q", vals[1], "world")
		}
	})

	t.Run("three strings different lengths", func(t *T) {
		original := []any{"a", "bb", "ccc"}
		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if vals[0] != "a" {
			t.Errorf("vals[0] = %q, want %q", vals[0], "a")
		}
		if vals[1] != "bb" {
			t.Errorf("vals[1] = %q, want %q", vals[1], "bb")
		}
		if vals[2] != "ccc" {
			t.Errorf("vals[2] = %q, want %q", vals[2], "ccc")
		}
	})

	t.Run("string and []byte", func(t *T) {
		original := []any{"hello", []byte{1, 2, 3}}
		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf([]byte{}),
		}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if vals[0] != "hello" {
			t.Errorf("vals[0] = %q, want %q", vals[0], "hello")
		}
		if !bytes.Equal(vals[1].([]byte), []byte{1, 2, 3}) {
			t.Errorf("vals[1] = %v, want %v", vals[1], []byte{1, 2, 3})
		}
	})

	t.Run("int32 and two strings", func(t *T) {
		original := []any{int32(42), "foo", "bar"}
		types := []reflect.Type{
			reflect.TypeOf(int32(0)),
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if vals[0] != int32(42) {
			t.Errorf("vals[0] = %v, want 42", vals[0])
		}
		if vals[1] != "foo" {
			t.Errorf("vals[1] = %q, want %q", vals[1], "foo")
		}
		if vals[2] != "bar" {
			t.Errorf("vals[2] = %q, want %q", vals[2], "bar")
		}
	})

	t.Run("large strings", func(t *T) {
		largeStr1 := make([]byte, 1000)
		largeStr2 := make([]byte, 500)
		for i := range largeStr1 {
			largeStr1[i] = byte(i % 256)
		}
		for i := range largeStr2 {
			largeStr2[i] = byte((i + 100) % 256)
		}

		original := []any{string(largeStr1), string(largeStr2)}
		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if vals[0] != string(largeStr1) {
			t.Errorf("large string 1 mismatch: len %d vs %d",
				len(vals[0].(string)), len(largeStr1))
		}
		if vals[1] != string(largeStr2) {
			t.Errorf("large string 2 mismatch: len %d vs %d",
				len(vals[1].(string)), len(largeStr2))
		}
	})

	t.Run("all empty strings", func(t *T) {
		original := []any{"", "", ""}
		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		for i, v := range vals {
			if v != "" {
				t.Errorf("vals[%d] = %q, want empty", i, v)
			}
		}
	})

	t.Run("complex mixed types", func(t *T) {
		original := []any{
			int32(123),
			true,
			"test string",
			int64(-999),
			[]byte{10, 20, 30},
			float64(3.14159),
		}
		types := []reflect.Type{
			reflect.TypeOf(int32(0)),
			reflect.TypeOf(true),
			reflect.TypeOf(""),
			reflect.TypeOf(int64(0)),
			reflect.TypeOf([]byte{}),
			reflect.TypeOf(float64(0)),
		}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if vals[0] != int32(123) {
			t.Errorf("int32: got %v, want 123", vals[0])
		}
		if vals[1] != true {
			t.Errorf("bool: got %v, want true", vals[1])
		}
		if vals[2] != "test string" {
			t.Errorf("string: got %q, want %q", vals[2], "test string")
		}
		if vals[3] != int64(-999) {
			t.Errorf("int64: got %v, want -999", vals[3])
		}
		if !bytes.Equal(vals[4].([]byte), []byte{10, 20, 30}) {
			t.Errorf("[]byte: got %v, want [10 20 30]", vals[4])
		}
		if vals[5] != 3.14159 {
			t.Errorf("float64: got %v, want 3.14159", vals[5])
		}
	})

	t.Run("all fixed types", func(t *T) {
		original := []any{
			true,
			int8(-1),
			int16(1000),
			int32(100000),
			int64(10000000000),
			uint8(255),
			uint16(65535),
			uint32(4000000000),
			uint64(10000000000000000000),
			float32(3.14),
			float64(2.71828),
		}
		types := []reflect.Type{
			reflect.TypeOf(true),
			reflect.TypeOf(int8(0)),
			reflect.TypeOf(int16(0)),
			reflect.TypeOf(int32(0)),
			reflect.TypeOf(int64(0)),
			reflect.TypeOf(uint8(0)),
			reflect.TypeOf(uint16(0)),
			reflect.TypeOf(uint32(0)),
			reflect.TypeOf(uint64(0)),
			reflect.TypeOf(float32(0)),
			reflect.TypeOf(float64(0)),
		}

		data := serializeLibfuzzerBytes(original)
		vals := deserializeLibfuzzerBytes(data, types)

		if vals[0] != true {
			t.Errorf("bool: got %v, want true", vals[0])
		}
		if vals[1] != int8(-1) {
			t.Errorf("int8: got %v, want -1", vals[1])
		}
		if vals[2] != int16(1000) {
			t.Errorf("int16: got %v, want 1000", vals[2])
		}
		if vals[3] != int32(100000) {
			t.Errorf("int32: got %v, want 100000", vals[3])
		}
		if vals[4] != int64(10000000000) {
			t.Errorf("int64: got %v, want 10000000000", vals[4])
		}
		if vals[5] != uint8(255) {
			t.Errorf("uint8: got %v, want 255", vals[5])
		}
		if vals[6] != uint16(65535) {
			t.Errorf("uint16: got %v, want 65535", vals[6])
		}
		if vals[7] != uint32(4000000000) {
			t.Errorf("uint32: got %v, want 4000000000", vals[7])
		}
		if vals[8] != uint64(10000000000000000000) {
			t.Errorf("uint64: got %v, want 10000000000000000000", vals[8])
		}
		if vals[9] != float32(3.14) {
			t.Errorf("float32: got %v, want 3.14", vals[9])
		}
		if vals[10] != float64(2.71828) {
			t.Errorf("float64: got %v, want 2.71828", vals[10])
		}
	})
}

// Safety tests (edge cases, short inputs)
func TestSafetyWithShortInputs(t *T) {
	t.Run("very short input for two strings", func(t *T) {
		data := []byte{1, 2, 3}
		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}
		vals := deserializeLibfuzzerBytes(data, types)

		if len(vals) != 2 {
			t.Fatalf("got %d vals, want 2", len(vals))
		}
	})

	t.Run("empty input for three strings", func(t *T) {
		data := []byte{}
		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}
		vals := deserializeLibfuzzerBytes(data, types)

		if len(vals) != 3 {
			t.Fatalf("got %d vals, want 3", len(vals))
		}
		for i, v := range vals {
			if v != "" {
				t.Errorf("vals[%d] = %q, want empty", i, v)
			}
		}
	})

	t.Run("weights only no data", func(t *T) {
		var data bytes.Buffer
		binary.Write(&data, binary.BigEndian, uint32(100))
		binary.Write(&data, binary.BigEndian, uint32(200))

		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}
		vals := deserializeLibfuzzerBytes(data.Bytes(), types)

		if len(vals) != 2 {
			t.Fatalf("got %d vals, want 2", len(vals))
		}
		if vals[0] != "" {
			t.Errorf("vals[0] = %q, want empty", vals[0])
		}
		if vals[1] != "" {
			t.Errorf("vals[1] = %q, want empty", vals[1])
		}
	})

	t.Run("all zero weights", func(t *T) {
		var data bytes.Buffer
		binary.Write(&data, binary.BigEndian, uint32(0))
		binary.Write(&data, binary.BigEndian, uint32(0))
		data.WriteString("abcdef")

		types := []reflect.Type{
			reflect.TypeOf(""),
			reflect.TypeOf(""),
		}
		vals := deserializeLibfuzzerBytes(data.Bytes(), types)

		if len(vals) != 2 {
			t.Fatalf("got %d vals, want 2", len(vals))
		}
		totalLen := len(vals[0].(string)) + len(vals[1].(string))
		if totalLen != 6 {
			t.Errorf("total length = %d, want 6", totalLen)
		}
	})

	t.Run("short input for fixed type", func(t *T) {
		data := []byte{0x01, 0x02}
		types := []reflect.Type{reflect.TypeOf(int32(0))}
		vals := deserializeLibfuzzerBytes(data, types)

		if len(vals) != 1 {
			t.Fatalf("got %d vals, want 1", len(vals))
		}
		if vals[0] != int32(0x01020000) {
			t.Errorf("vals[0] = %v, want %v", vals[0], int32(0x01020000))
		}
	})
}

// writeCorpusFile tests
func TestWriteCorpusFile(t *T) {
	t.Run("writes file with hash name", func(t *T) {
		tmpDir, err := os.MkdirTemp("", "corpus_test")
		if err != nil {
			t.Fatalf("failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tmpDir)

		data := []byte("test corpus data")
		err = writeCorpusFile(tmpDir, data)
		if err != nil {
			t.Fatalf("writeCorpusFile failed: %v", err)
		}

		entries, err := os.ReadDir(tmpDir)
		if err != nil {
			t.Fatalf("failed to read dir: %v", err)
		}
		if len(entries) != 1 {
			t.Fatalf("expected 1 file, got %d", len(entries))
		}

		p := filepath.Join(tmpDir, entries[0].Name())
		content, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("failed to read file: %v", err)
		}
		if !bytes.Equal(content, data) {
			t.Errorf("file content mismatch")
		}
	})

	t.Run("skips duplicate content", func(t *T) {
		tmpDir, err := os.MkdirTemp("", "corpus_test")
		if err != nil {
			t.Fatalf("failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tmpDir)

		data := []byte("duplicate data")

		err = writeCorpusFile(tmpDir, data)
		if err != nil {
			t.Fatalf("first write failed: %v", err)
		}
		err = writeCorpusFile(tmpDir, data)
		if err != nil {
			t.Fatalf("second write failed: %v", err)
		}

		entries, err := os.ReadDir(tmpDir)
		if err != nil {
			t.Fatalf("failed to read dir: %v", err)
		}
		if len(entries) != 1 {
			t.Errorf("expected 1 file (deduped), got %d", len(entries))
		}
	})
}

// runFuzzIteration tests
// testFuzzState creates a fuzz state for testing
func testFuzzState() *fuzzState {
	return &fuzzState{
		mode: libFuzzerMode,
	}
}

func TestRunFuzzIteration(t *T) {
	called := ""
	targets := []InternalFuzzTarget{
		{Name: "FuzzA", Fn: func(f *F) { called = "A" }},
		{Name: "FuzzB", Fn: func(f *F) { called = "B" }},
		{Name: "FuzzC", Fn: func(f *F) { called = "C" }},
	}

	tests := []struct {
		name       string
		targetName string
		wantCalled string
	}{
		{"exact match", "FuzzB", "B"},
		{"first target", "FuzzA", "A"},
		{"last target", "FuzzC", "C"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *T) {
			called = ""
			var target *InternalFuzzTarget
			for i := range targets {
				if targets[i].Name == tt.targetName {
					target = &targets[i]
					break
				}
			}
			if target != nil {
				runFuzzIteration(target.Name, target.Fn, testFuzzState(), []byte("test"))
			}
			if called != tt.wantCalled {
				t.Errorf("called = %q, want %q", called, tt.wantCalled)
			}
		})
	}
}

func TestRunFuzzIterationFLevelMethods(t *T) {
	t.Run("f.Skip stops execution", func(t *T) {
		afterSkipCalled := false
		fn := func(f *F) {
			f.Skip("test skip")
			afterSkipCalled = true
		}

		runFuzzIteration("FuzzSkip", fn, testFuzzState(), []byte("test"))

		if afterSkipCalled {
			t.Error("code after f.Skip() was executed")
		}
	})

	t.Run("f.SkipNow stops execution", func(t *T) {
		afterSkipNowCalled := false
		fn := func(f *F) {
			f.SkipNow()
			afterSkipNowCalled = true
		}

		runFuzzIteration("FuzzSkipNow", fn, testFuzzState(), []byte("test"))

		if afterSkipNowCalled {
			t.Error("code after f.SkipNow() was executed")
		}
	})

	t.Run("f.Fatal causes panic", func(t *T) {
		fn := func(f *F) {
			f.Fatal("test fatal")
		}

		defer func() {
			r := recover()
			if r == nil {
				t.Error("f.Fatal() did not cause panic")
			}
			s, ok := r.(string)
			if !ok || s != "testing: fuzz test failed before Fuzz() was called" {
				t.Errorf("unexpected panic value: %v", r)
			}
		}()

		runFuzzIteration("FuzzFatal", fn, testFuzzState(), []byte("test"))
		t.Error("should not reach here")
	})

	t.Run("f.FailNow causes panic", func(t *T) {
		fn := func(f *F) {
			f.FailNow()
		}

		defer func() {
			r := recover()
			if r == nil {
				t.Error("f.FailNow() did not cause panic")
			}
		}()

		runFuzzIteration("FuzzFailNow", fn, testFuzzState(), []byte("test"))
		t.Error("should not reach here")
	})

	t.Run("f.Cleanup is called", func(t *T) {
		cleanupCalled := false
		fn := func(f *F) {
			f.Cleanup(func() {
				cleanupCalled = true
			})
		}

		runFuzzIteration("FuzzCleanup", fn, testFuzzState(), []byte("test"))

		if !cleanupCalled {
			t.Error("f.Cleanup() was not called")
		}
	})

	t.Run("f.Cleanup order is LIFO", func(t *T) {
		var order []int
		fn := func(f *F) {
			f.Cleanup(func() { order = append(order, 1) })
			f.Cleanup(func() { order = append(order, 2) })
			f.Cleanup(func() { order = append(order, 3) })
		}

		runFuzzIteration("FuzzCleanupOrder", fn, testFuzzState(), []byte("test"))

		if len(order) != 3 || order[0] != 3 || order[1] != 2 || order[2] != 1 {
			t.Errorf("cleanup order = %v, want [3 2 1]", order)
		}
	})

	t.Run("f.Error marks failed but continues execution", func(t *T) {
		afterErrorCalled := false
		fn := func(f *F) {
			f.Error("test error")
			afterErrorCalled = true
		}

		defer func() {
			r := recover()
			if r == nil {
				t.Error("expected panic after f.Error() at F level")
			}
			if !afterErrorCalled {
				t.Error("code after f.Error() was not executed")
			}
		}()

		runFuzzIteration("FuzzError", fn, testFuzzState(), []byte("test"))
		t.Error("should not reach here")
	})

	t.Run("f.Log does not affect execution", func(t *T) {
		afterLogCalled := false
		fn := func(f *F) {
			f.Log("test log")
			afterLogCalled = true
		}

		runFuzzIteration("FuzzLog", fn, testFuzzState(), []byte("test"))

		if !afterLogCalled {
			t.Error("code after f.Log() was not executed")
		}
	})

	t.Run("f.Name returns correct name", func(t *T) {
		var gotName string
		fn := func(f *F) {
			gotName = f.Name()
		}

		runFuzzIteration("FuzzTestName", fn, testFuzzState(), []byte("test"))

		if gotName != "FuzzTestName" {
			t.Errorf("f.Name() = %q, want %q", gotName, "FuzzTestName")
		}
	})

	t.Run("f.TempDir creates directory", func(t *T) {
		var tempDir string
		fn := func(f *F) {
			tempDir = f.TempDir()
		}

		runFuzzIteration("FuzzTempDir", fn, testFuzzState(), []byte("test"))

		if tempDir == "" {
			t.Error("f.TempDir() returned empty string")
		}
	})

	t.Run("f.Skipped returns correct value", func(t *T) {
		var skippedBefore bool
		fn := func(f *F) {
			skippedBefore = f.Skipped()
			f.Skip("skip")
		}

		runFuzzIteration("FuzzSkipped", fn, testFuzzState(), []byte("test"))

		if skippedBefore {
			t.Error("f.Skipped() returned true before skip")
		}
	})
}

func TestFLevelSetenvRestoration(t *T) {
	t.Run("restores existing variable", func(t *T) {
		os.Setenv("TEST_LIBFUZZER_VAR", "original")
		defer os.Unsetenv("TEST_LIBFUZZER_VAR")

		values := []string{}
		fn := func(f *F) {
			values = append(values, os.Getenv("TEST_LIBFUZZER_VAR"))
			f.Setenv("TEST_LIBFUZZER_VAR", "changed")
			values = append(values, os.Getenv("TEST_LIBFUZZER_VAR"))
		}

		runFuzzIteration("FuzzSetenv1", fn, testFuzzState(), []byte("test"))
		runFuzzIteration("FuzzSetenv2", fn, testFuzzState(), []byte("test"))

		if len(values) != 4 {
			t.Errorf("got %d values, want 4", len(values))
			return
		}
		if values[0] != "original" || values[2] != "original" {
			t.Errorf("values before Setenv = %v, want [original, _, original, _]", values)
		}
		if values[1] != "changed" || values[3] != "changed" {
			t.Errorf("values after Setenv = %v, want [_, changed, _, changed]", values)
		}
	})

	t.Run("restores unset variable", func(t *T) {
		os.Unsetenv("TEST_LIBFUZZER_NEW")

		fn := func(f *F) {
			f.Setenv("TEST_LIBFUZZER_NEW", "value")
		}

		runFuzzIteration("FuzzSetenvNew", fn, testFuzzState(), []byte("test"))

		if v := os.Getenv("TEST_LIBFUZZER_NEW"); v != "" {
			t.Errorf("variable not restored: got %q, want empty", v)
		}
	})
}

func TestFLevelChdirRestoration(t *T) {
	originalDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get working directory: %v", err)
	}

	t.Run("restores original directory", func(t *T) {
		dirs := []string{}
		fn := func(f *F) {
			d, _ := os.Getwd()
			dirs = append(dirs, d)
			tmpDir := f.TempDir()
			f.Chdir(tmpDir)
			d, _ = os.Getwd()
			dirs = append(dirs, d)
		}

		runFuzzIteration("FuzzChdir1", fn, testFuzzState(), []byte("test"))
		runFuzzIteration("FuzzChdir2", fn, testFuzzState(), []byte("test"))

		if len(dirs) != 4 {
			t.Errorf("got %d dirs, want 4", len(dirs))
			return
		}
		if dirs[0] != originalDir || dirs[2] != originalDir {
			t.Errorf("dirs before Chdir = [%s, _, %s, _], want [%s, _, %s, _]",
				dirs[0], dirs[2], originalDir, originalDir)
		}
		if dirs[1] == originalDir || dirs[3] == originalDir {
			t.Error("Chdir didn't change to temp directory")
		}
	})

	currentDir, _ := os.Getwd()
	if currentDir != originalDir {
		t.Errorf("not restored to original dir: got %s, want %s", currentDir, originalDir)
		os.Chdir(originalDir)
	}
}

// Full round-trip fuzzer helpers and types
var primitiveTypes = []reflect.Type{
	reflect.TypeOf(true),
	reflect.TypeOf(int8(0)),
	reflect.TypeOf(int16(0)),
	reflect.TypeOf(int32(0)),
	reflect.TypeOf(int64(0)),
	reflect.TypeOf(uint8(0)),
	reflect.TypeOf(uint16(0)),
	reflect.TypeOf(uint32(0)),
	reflect.TypeOf(uint64(0)),
	reflect.TypeOf(float32(0)),
	reflect.TypeOf(float64(0)),
	reflect.TypeOf(""),
	reflect.TypeOf([]byte{}),
}

var fixedSizeTypes = []reflect.Type{
	reflect.TypeOf(true),
	reflect.TypeOf(int8(0)),
	reflect.TypeOf(int16(0)),
	reflect.TypeOf(int32(0)),
	reflect.TypeOf(int64(0)),
	reflect.TypeOf(uint8(0)),
	reflect.TypeOf(uint16(0)),
	reflect.TypeOf(uint32(0)),
	reflect.TypeOf(uint64(0)),
	reflect.TypeOf(float32(0)),
	reflect.TypeOf(float64(0)),
}

func countDynamicArgs(types []reflect.Type) int {
	count := 0
	for _, t := range types {
		isString := t.Kind() == reflect.String
		isByteSlice := t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8
		if isString || isByteSlice {
			count++
		}
	}
	return count
}

func getType(index byte, typeList []reflect.Type) reflect.Type {
	return typeList[int(index)%len(typeList)]
}

func extractTypesFromInput(
	input []byte, numArgs int, typeList []reflect.Type,
) []reflect.Type {
	types := make([]reflect.Type, numArgs)
	for i := 0; i < numArgs; i++ {
		types[i] = getType(input[1+i], typeList)
	}
	return types
}

type dynamicContent struct {
	totalLength int
	data        []byte
}

func collectDynamicContent(vals []any, types []reflect.Type) dynamicContent {
	var content dynamicContent
	for i, t := range types {
		switch t.Kind() {
		case reflect.String:
			s := vals[i].(string)
			content.totalLength += len(s)
			content.data = append(content.data, s...)
		case reflect.Slice:
			if t.Elem().Kind() == reflect.Uint8 {
				b := vals[i].([]byte)
				content.totalLength += len(b)
				content.data = append(content.data, b...)
			}
		}
	}
	return content
}

func verifyFixedTypesMatch(t *T, types []reflect.Type, vals1, vals2 []any) {
	for i := range types {
		if fixedTypeSize(types[i]) > 0 {
			if !valuesEqual(vals1[i], vals2[i]) {
				t.Errorf("fixed value[%d] mismatch: got %v, want %v", i, vals2[i], vals1[i])
			}
		}
	}
}

func verifyDynamicContentPreserved(t *T, types []reflect.Type, vals1, vals3 []any) {
	content1 := collectDynamicContent(vals1, types)
	content3 := collectDynamicContent(vals3, types)

	if content1.totalLength != content3.totalLength {
		t.Errorf("dynamic data total mismatch: got %d, want %d",
			content3.totalLength, content1.totalLength)
	}
	if !bytes.Equal(content1.data, content3.data) {
		t.Errorf("dynamic data content mismatch")
	}
}

func valuesEqual(a, b any) bool {
	switch av := a.(type) {
	case []byte:
		bv, ok := b.([]byte)
		if !ok {
			return false
		}
		return bytes.Equal(av, bv)
	case float32:
		bv, ok := b.(float32)
		if !ok {
			return false
		}
		if math.IsNaN(float64(av)) && math.IsNaN(float64(bv)) {
			return math.Float32bits(av) == math.Float32bits(bv)
		}
		return av == bv
	case float64:
		bv, ok := b.(float64)
		if !ok {
			return false
		}
		if math.IsNaN(av) && math.IsNaN(bv) {
			return math.Float64bits(av) == math.Float64bits(bv)
		}
		return av == bv
	default:
		return a == b
	}
}

// Full round-trip fuzzers
func FuzzFullCorpusRoundTrip(f *F) {
	f.Fuzz(func(t *T, input []byte) {
		if len(input) < 2 {
			return
		}

		numArgs := int(input[0]%30) + 1
		if len(input) < 1+numArgs {
			return
		}

		types := extractTypesFromInput(input, numArgs, primitiveTypes)
		dataStart := 1 + numArgs
		data := input[dataStart:]

		vals1 := deserializeLibfuzzerBytes(data, types)
		if len(vals1) != numArgs {
			t.Fatalf("deserialize returned %d values, want %d", len(vals1), numArgs)
		}

		corpusBytes := marshalGoCorpusFile(vals1)

		vals2, err := parseGoCorpusFile(corpusBytes, types)
		if err != nil {
			t.Fatalf("parseGoCorpusFile failed: %v\ncorpus:\n%s", err, string(corpusBytes))
		}

		for i := range vals1 {
			if !valuesEqual(vals1[i], vals2[i]) {
				t.Errorf("corpus round-trip mismatch at [%d]: got %v (%T), want %v (%T)",
					i, vals2[i], vals2[i], vals1[i], vals1[i])
			}
		}

		data2 := serializeLibfuzzerBytes(vals2)
		vals3 := deserializeLibfuzzerBytes(data2, types)
		if len(vals3) != numArgs {
			t.Fatalf("second deserialize returned %d values, want %d", len(vals3), numArgs)
		}

		numDynamic := countDynamicArgs(types)
		if numDynamic <= 1 {
			for i := range vals1 {
				if !valuesEqual(vals1[i], vals3[i]) {
					t.Errorf("full round-trip mismatch at [%d]: got %v (%T), want %v (%T)",
						i, vals3[i], vals3[i], vals1[i], vals1[i])
				}
			}
		} else {
			verifyFixedTypesMatch(t, types, vals1, vals3)
			verifyDynamicContentPreserved(t, types, vals1, vals3)
		}
	})
}

func FuzzFullCorpusRoundTripFixedOnly(f *F) {
	f.Fuzz(func(t *T, input []byte) {
		if len(input) < 2 {
			return
		}

		numArgs := int(input[0]%30) + 1
		if len(input) < 1+numArgs {
			return
		}

		types := extractTypesFromInput(input, numArgs, fixedSizeTypes)
		data := input[1+numArgs:]

		vals1 := deserializeLibfuzzerBytes(data, types)
		corpusBytes := marshalGoCorpusFile(vals1)

		vals2, err := parseGoCorpusFile(corpusBytes, types)
		if err != nil {
			t.Fatalf("parseGoCorpusFile failed: %v", err)
		}

		for i := range vals1 {
			if !valuesEqual(vals1[i], vals2[i]) {
				t.Errorf("corpus round-trip[%d]: got %v, want %v", i, vals2[i], vals1[i])
			}
		}

		data2 := serializeLibfuzzerBytes(vals2)
		vals3 := deserializeLibfuzzerBytes(data2, types)

		for i := range vals1 {
			if !valuesEqual(vals1[i], vals3[i]) {
				t.Errorf("full round-trip[%d]: got %v (%T), want %v (%T)",
					i, vals3[i], vals3[i], vals1[i], vals1[i])
			}
		}
	})
}
