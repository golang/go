// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalaropt

import (
	"math/bits"
	"testing"
)

const descriptorTag uint8 = 37

type descriptorRecord struct {
	offset, advance, payload uint64
}

// The address is a sum of aligned byte offsets, with a tag in its low byte.
// Wrapping the address preserves those eight zero bits before adding the tag.
func packDescriptor(offset, advance uint64, tag uint8) uint64 {
	return (offset &^ 255) + (advance &^ 255) + uint64(tag)
}

func consumeDescriptor(header, payload uint64) uint64 {
	if uint8(header) == descriptorTag {
		return (header >> 8) ^ payload
	}
	return header ^ bits.RotateLeft64(payload, 7)
}

//go:noinline
func packedDescriptors(records []descriptorRecord) uint64 {
	var checksum uint64
	for _, r := range records {
		header := packDescriptor(r.offset, r.advance, descriptorTag)
		checksum += consumeDescriptor(header, r.payload)
	}
	return checksum
}

//go:noinline
func correlatedDescriptors(records []descriptorRecord) uint64 {
	var checksum uint64
	for _, r := range records {
		var x, y uint64
		if r.offset&1 != 0 {
			x, y = 1, 3
		} else {
			x, y = 2, 2
		}
		checksum += (x + y) ^ r.payload
	}
	return checksum
}

type headerRecord struct {
	key      uint32
	sequence uint16
	payload  uint64
}

// The native form occupies 48 bits. Rotating it puts the reserved 16 bits
// at the bottom of the wire form, where the shared consumer checks them.
func packHeader(key uint32, sequence uint16) uint64 {
	native := uint64(key)<<16 | uint64(sequence)
	return bits.RotateLeft64(native, 16)
}

func consumeHeader(wire, payload uint64) uint64 {
	if uint16(wire) == 0 {
		return (wire >> 16) ^ payload
	}
	return wire ^ bits.RotateLeft64(payload, 7)
}

//go:noinline
func packedHeaders(records []headerRecord) uint64 {
	var checksum uint64
	for _, r := range records {
		wire := packHeader(r.key, r.sequence)
		checksum += consumeHeader(wire, r.payload)
	}
	return checksum
}

func TestScalarProcessingLoopsDescriptors(t *testing.T) {
	values := []uint64{0, 1, 255, 256, 257, 1 << 63, ^uint64(0)}
	var extremes []descriptorRecord
	for _, offset := range values {
		for _, advance := range values {
			for _, payload := range values {
				extremes = append(extremes, descriptorRecord{offset, advance, payload})
			}
		}
	}
	for _, tc := range []struct {
		name    string
		records []descriptorRecord
	}{
		{"nil", nil},
		{"empty", []descriptorRecord{}},
		{"singleton", []descriptorRecord{{0x1234, 0x2345, 0x3456}}},
		{"extremes", extremes},
		{"wrapping", []descriptorRecord{{^uint64(0), ^uint64(0), 0}, {0, 0, ^uint64(0)}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var want uint64
			var wantCorrelated uint64
			for _, r := range tc.records {
				blocks := ((r.offset >> 8) + (r.advance >> 8)) & ((1 << 56) - 1)
				want += blocks ^ r.payload
				wantCorrelated += r.payload ^ 4
			}
			if got := packedDescriptors(tc.records); got != want {
				t.Fatalf("descriptor checksum = %#x, want %#x", got, want)
			}
			if got := correlatedDescriptors(tc.records); got != wantCorrelated {
				t.Fatalf("correlated checksum = %#x, want %#x", got, wantCorrelated)
			}
		})
	}
	original := descriptorRecord{0x1234, 0x2345, 0x3456}
	want := packedDescriptors([]descriptorRecord{original})
	for _, changed := range []descriptorRecord{
		{original.offset + 256, original.advance, original.payload},
		{original.offset, original.advance + 256, original.payload},
		{original.offset, original.advance, original.payload ^ 1},
	} {
		blocks := ((changed.offset >> 8) + (changed.advance >> 8)) & ((1 << 56) - 1)
		if got := packedDescriptors([]descriptorRecord{changed}); got != blocks^changed.payload || got == want {
			t.Fatalf("descriptor fields stopped affecting the checksum: %+v -> %#x", changed, got)
		}
	}
	for _, payload := range []uint64{0, 1, original.payload, ^uint64(0)} {
		changed := original
		changed.payload = payload
		if got := correlatedDescriptors([]descriptorRecord{changed}); got != payload^4 {
			t.Fatalf("correlated payload %#x = %#x, want %#x", payload, got, payload^4)
		}
	}
}

func TestScalarProcessingLoopsHeaders(t *testing.T) {
	var extremes []headerRecord
	for _, key := range []uint32{0, 1, 1 << 31, ^uint32(0)} {
		for _, sequence := range []uint16{0, 1, 1 << 15, ^uint16(0)} {
			for _, payload := range []uint64{0, 1, 1 << 63, ^uint64(0)} {
				extremes = append(extremes, headerRecord{key, sequence, payload})
			}
		}
	}
	for _, tc := range []struct {
		name    string
		records []headerRecord
	}{
		{"nil", nil},
		{"empty", []headerRecord{}},
		{"singleton", []headerRecord{{0x12345678, 0x9abc, 0x3456}}},
		{"extremes", extremes},
		{"wrapping", []headerRecord{{^uint32(0), ^uint16(0), 0}, {0, 0, ^uint64(0)}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var want uint64
			for _, r := range tc.records {
				want += (uint64(r.key)<<16 | uint64(r.sequence)) ^ r.payload
			}
			if got := packedHeaders(tc.records); got != want {
				t.Fatalf("header checksum = %#x, want %#x", got, want)
			}
		})
	}
	original := headerRecord{0x12345678, 0x9abc, 0x3456}
	want := packedHeaders([]headerRecord{original})
	for _, changed := range []headerRecord{
		{original.key + 1, original.sequence, original.payload},
		{original.key, original.sequence + 1, original.payload},
		{original.key, original.sequence, original.payload ^ 1},
	} {
		raw := uint64(changed.key)<<16 | uint64(changed.sequence)
		if got := packedHeaders([]headerRecord{changed}); got != raw^changed.payload || got == want {
			t.Fatalf("header fields stopped affecting the checksum: %+v -> %#x", changed, got)
		}
	}
}

func TestScalarProcessingLoopsGenericDispatch(t *testing.T) {
	for _, payload := range []uint64{0, 0x123456789abcdef, ^uint64(0)} {
		for _, tag := range []uint8{0, 36, descriptorTag, 38, 255} {
			offset, advance := uint64(0xfedcba9876543210), uint64(0x123456789abcdef0)
			blocks := ((offset >> 8) + (advance >> 8)) & ((1 << 56) - 1)
			header := packDescriptor(offset, advance, tag)
			if want := blocks<<8 | uint64(tag); header != want {
				t.Fatalf("pack tag %d = %#x, want %#x", tag, header, want)
			}
			want := header ^ bits.RotateLeft64(payload, 7)
			if tag == descriptorTag {
				want = blocks ^ payload
			}
			if got := consumeDescriptor(header, payload); got != want {
				t.Fatalf("consume tag %d payload %#x = %#x, want %#x", tag, payload, got, want)
			}
		}
		for _, reserved := range []uint16{0, 1, 0x8000, 0xffff} {
			wire := packHeader(0x12345678, 0x9abc) | uint64(reserved)
			want := wire ^ bits.RotateLeft64(payload, 7)
			if reserved == 0 {
				want = (uint64(0x12345678)<<16 | 0x9abc) ^ payload
			}
			if got := consumeHeader(wire, payload); got != want {
				t.Fatalf("consume reserved %#x payload %#x = %#x, want %#x", reserved, payload, got, want)
			}
		}
	}
}

func BenchmarkPackedDescriptors(b *testing.B) {
	records := make([]descriptorRecord, 1024)
	for i := range records {
		x := uint64(i+1) * 0x9e3779b97f4a7c15
		records[i] = descriptorRecord{x, bits.RotateLeft64(x, 23), ^x}
	}
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += packedDescriptors(records)
	}
	sink = checksum
}

func BenchmarkPackedHeaders(b *testing.B) {
	records := make([]headerRecord, 1024)
	for i := range records {
		x := uint64(i+1) * 0x9e3779b97f4a7c15
		records[i] = headerRecord{uint32(x >> 16), uint16(x), ^x}
	}
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += packedHeaders(records)
	}
	sink = checksum
}

func BenchmarkCorrelatedDescriptors(b *testing.B) {
	records := make([]descriptorRecord, 1024)
	for i := range records {
		x := uint64(i+1) * 0x9e3779b97f4a7c15
		records[i] = descriptorRecord{x, bits.RotateLeft64(x, 23), ^x}
	}
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += correlatedDescriptors(records)
	}
	sink = checksum
}
