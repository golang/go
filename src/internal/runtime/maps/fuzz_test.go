// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package maps implements Go's builtin map type.
package maps_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"internal/runtime/maps"
	"reflect"
	"testing"
	"unsafe"
)

// The input to FuzzTable is a binary-encoded array of fuzzCommand structs.
//
// Each fuzz call begins with an empty Map[uint16, uint32].
//
// Each command is then executed on the map in sequence. Operations with
// output (e.g., Get) are verified against a reference map.
type fuzzCommand struct {
	Op fuzzOp

	// Used for Get, Put, Delete.
	Key uint16

	// Used for Put.
	Elem uint32
}

// Encoded size of fuzzCommand.
var fuzzCommandSize = binary.Size(fuzzCommand{})

type fuzzOp uint8

const (
	fuzzOpGet fuzzOp = iota
	fuzzOpPut
	fuzzOpDelete
)

func encode(fc []fuzzCommand) []byte {
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, fc); err != nil {
		panic(fmt.Sprintf("error writing %v: %v", fc, err))
	}
	return buf.Bytes()
}

func decode(b []byte) []fuzzCommand {
	// Round b down to a multiple of fuzzCommand size. i.e., ignore extra
	// bytes of input.
	entries := len(b) / fuzzCommandSize
	usefulSize := entries * fuzzCommandSize
	b = b[:usefulSize]

	fc := make([]fuzzCommand, entries)
	buf := bytes.NewReader(b)
	if err := binary.Read(buf, binary.LittleEndian, &fc); err != nil {
		panic(fmt.Sprintf("error reading %v: %v", b, err))
	}

	return fc
}

func TestEncodeDecode(t *testing.T) {
	fc := []fuzzCommand{
		{
			Op:   fuzzOpPut,
			Key:  123,
			Elem: 456,
		},
		{
			Op:  fuzzOpGet,
			Key: 123,
		},
	}

	b := encode(fc)
	got := decode(b)
	if !reflect.DeepEqual(fc, got) {
		t.Errorf("encode-decode roundtrip got %+v want %+v", got, fc)
	}

	// Extra trailing bytes ignored.
	b = append(b, 42)
	got = decode(b)
	if !reflect.DeepEqual(fc, got) {
		t.Errorf("encode-decode (extra byte) roundtrip got %+v want %+v", got, fc)
	}
}

func FuzzTable(f *testing.F) {
	// All of the ops.
	f.Add(encode([]fuzzCommand{
		{
			Op:   fuzzOpPut,
			Key:  123,
			Elem: 456,
		},
		{
			Op:  fuzzOpDelete,
			Key: 123,
		},
		{
			Op:  fuzzOpGet,
			Key: 123,
		},
	}))

	// Add enough times to trigger grow.
	f.Add(encode([]fuzzCommand{
		{
			Op:   fuzzOpPut,
			Key:  1,
			Elem: 101,
		},
		{
			Op:   fuzzOpPut,
			Key:  2,
			Elem: 102,
		},
		{
			Op:   fuzzOpPut,
			Key:  3,
			Elem: 103,
		},
		{
			Op:   fuzzOpPut,
			Key:  4,
			Elem: 104,
		},
		{
			Op:   fuzzOpPut,
			Key:  5,
			Elem: 105,
		},
		{
			Op:   fuzzOpPut,
			Key:  6,
			Elem: 106,
		},
		{
			Op:   fuzzOpPut,
			Key:  7,
			Elem: 107,
		},
		{
			Op:   fuzzOpPut,
			Key:  8,
			Elem: 108,
		},
		{
			Op:  fuzzOpGet,
			Key: 1,
		},
		{
			Op:  fuzzOpDelete,
			Key: 2,
		},
		{
			Op:   fuzzOpPut,
			Key:  2,
			Elem: 42,
		},
		{
			Op:  fuzzOpGet,
			Key: 2,
		},
	}))

	f.Fuzz(func(t *testing.T, in []byte) {
		fc := decode(in)
		if len(fc) == 0 {
			return
		}

		m, _ := maps.NewTestMap[uint16, uint32](8)
		ref := make(map[uint16]uint32)
		for _, c := range fc {
			switch c.Op {
			case fuzzOpGet:
				elemPtr, ok := m.Get(unsafe.Pointer(&c.Key))
				refElem, refOK := ref[c.Key]

				if ok != refOK {
					t.Errorf("Get(%d) got ok %v want ok %v", c.Key, ok, refOK)
				}
				if !ok {
					continue
				}
				gotElem := *(*uint32)(elemPtr)
				if gotElem != refElem {
					t.Errorf("Get(%d) got %d want %d", c.Key, gotElem, refElem)
				}
			case fuzzOpPut:
				m.Put(unsafe.Pointer(&c.Key), unsafe.Pointer(&c.Elem))
				ref[c.Key] = c.Elem
			case fuzzOpDelete:
				m.Delete(unsafe.Pointer(&c.Key))
				delete(ref, c.Key)
			default:
				// Just skip this command to keep the fuzzer
				// less constrained.
				continue
			}
		}
	})
}
