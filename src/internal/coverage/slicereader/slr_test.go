// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slicereader

import (
	"internal/binary"
	"io"
	"testing"
)

func TestSliceReader(t *testing.T) {
	b := []byte{}

	bt := make([]byte, 4)
	e32 := uint32(1030507)
	binary.LittleEndian.PutUint32(bt, e32)
	b = append(b, bt...)

	bt = make([]byte, 8)
	e64 := uint64(907050301)
	binary.LittleEndian.PutUint64(bt, e64)
	b = append(b, bt...)

	b = appendUleb128(b, uint(e32))
	b = appendUleb128(b, uint(e64))
	b = appendUleb128(b, 6)
	s1 := "foobar"
	s1b := []byte(s1)
	b = append(b, s1b...)
	b = appendUleb128(b, 9)
	s2 := "bazbasher"
	s2b := []byte(s2)
	b = append(b, s2b...)

	readStr := func(slr *Reader) string {
		len := slr.ReadULEB128()
		return slr.ReadString(int64(len))
	}

	for i := 0; i < 2; i++ {
		slr := NewReader(b, i == 0)
		g32 := slr.ReadUint32()
		if g32 != e32 {
			t.Fatalf("slr.ReadUint32() got %d want %d", g32, e32)
		}
		g64 := slr.ReadUint64()
		if g64 != e64 {
			t.Fatalf("slr.ReadUint64() got %d want %d", g64, e64)
		}
		g32 = uint32(slr.ReadULEB128())
		if g32 != e32 {
			t.Fatalf("slr.ReadULEB128() got %d want %d", g32, e32)
		}
		g64 = slr.ReadULEB128()
		if g64 != e64 {
			t.Fatalf("slr.ReadULEB128() got %d want %d", g64, e64)
		}
		gs1 := readStr(slr)
		if gs1 != s1 {
			t.Fatalf("readStr got %s want %s", gs1, s1)
		}
		gs2 := readStr(slr)
		if gs2 != s2 {
			t.Fatalf("readStr got %s want %s", gs2, s2)
		}
		if _, err := slr.Seek(4, io.SeekStart); err != nil {
			t.Fatal(err)
		}
		off := slr.Offset()
		if off != 4 {
			t.Fatalf("Offset() returned %d wanted 4", off)
		}
		g64 = slr.ReadUint64()
		if g64 != e64 {
			t.Fatalf("post-seek slr.ReadUint64() got %d want %d", g64, e64)
		}
	}
}

func appendUleb128(b []byte, v uint) []byte {
	for {
		c := uint8(v & 0x7f)
		v >>= 7
		if v != 0 {
			c |= 0x80
		}
		b = append(b, c)
		if c&0x80 == 0 {
			break
		}
	}
	return b
}
