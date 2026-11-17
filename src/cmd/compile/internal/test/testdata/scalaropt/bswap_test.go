// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalaropt

import (
	"math/bits"
	"testing"
)

// Byte-at-a-time reference, independent of the intrinsic and its lowering.
func bswapReference(x uint64, width uint) uint64 {
	var result uint64
	for shift := uint(0); shift < width; shift += 8 {
		result = result<<8 | (x>>shift)&255
	}
	return result
}

//go:noinline
func bswapValue16(x uint16) uint64 { return uint64(bits.ReverseBytes16(x)) }

//go:noinline
func bswapValue32(x uint32) uint32 { return bits.ReverseBytes32(x) }

//go:noinline
func bswapValue64(x uint64) uint64 { return bits.ReverseBytes64(x) }

//go:noinline
func bswapMaskedLoad(p *uint64) uint64 {
	return bits.ReverseBytes64(*p&0xffff) & 0x0000ffffffffffff
}

//go:noinline
func bswapMaskedIndex(words []uint64, i int) uint64 {
	return bits.ReverseBytes64(words[i]&0xffff) & 0x0000ffffffffffff
}

//go:noinline
func bswapMaskedDivision(x, y uint64) uint64 {
	return bits.ReverseBytes64((x/y)&0xffff) & 0x0000ffffffffffff
}

//go:noinline
func bswapMaskedCall(input func() uint64) uint64 {
	return bits.ReverseBytes64(input()&0xffff) & 0x0000ffffffffffff
}

// Only SCCP proves this multiplicative loop state constant. The payload
// loads and reduction remain linear work after the dispatch is removed.
//
//go:noinline
func bswapLateDispatch(words []uint64) uint64 {
	state := uint64(1)
	var sum uint64
	for _, word := range words {
		state *= state
		if bits.ReverseBytes64(state) == uint64(1)<<56 {
			sum += word
		} else {
			sum ^= word
		}
	}
	return sum
}

//go:noinline
func bswapDynamicDispatch(words []uint64, state uint64) uint64 {
	var sum uint64
	for _, word := range words {
		state *= state
		if bits.ReverseBytes64(state) == uint64(1)<<56 {
			sum += word
		} else {
			sum ^= word
		}
	}
	return sum
}

type bswapRecord struct {
	field, payload uint64
}

// The 16-bit field occupies the first two bytes of the wire value. Its
// reserved bytes are zero, while both the field and payload affect the sum.
//
//go:noinline
func bswapNetworkByteFields(records []bswapRecord) uint64 {
	var checksum uint64
	for _, record := range records {
		wire := bits.ReverseBytes64(record.field & 0xffff)
		if wire&0x0000ffffffffffff == 0 {
			checksum += (wire >> 48) ^ record.payload
		} else {
			checksum += wire ^ bits.RotateLeft64(record.payload, 7)
		}
	}
	return checksum
}

func TestScalarBswapParity(t *testing.T) {
	for i := uint64(0); i < 65536; i++ {
		x := i * 0x9e3779b97f4a7c15
		if got, want := bswapValue16(uint16(i)), bswapReference(i, 16); got != want {
			t.Fatalf("16-bit swap(%x) = %x, want %x", i, got, want)
		}
		if got, want := uint64(bswapValue32(uint32(x))), bswapReference(x, 32); got != want {
			t.Fatalf("32-bit swap(%x) = %x, want %x", x, got, want)
		}
		if got, want := bswapValue64(x), bswapReference(x, 64); got != want {
			t.Fatalf("64-bit swap(%x) = %x, want %x", x, got, want)
		}
	}
	for _, x := range []uint64{0, 1, 1 << 63, ^uint64(0)} {
		if got, want := bswapValue64(x), bswapReference(x, 64); got != want {
			t.Fatalf("extreme swap(%x) = %x, want %x", x, got, want)
		}
		if got := bswapMaskedLoad(&x); got != 0 {
			t.Fatalf("masked swap(%x) = %x, want zero", x, got)
		}
	}
}

func TestScalarBswapProcessing(t *testing.T) {
	var words [1024]uint64
	var records [1024]bswapRecord
	for i := range words {
		x := uint64(i+1) * 0x9e3779b97f4a7c15
		words[i] = x
		records[i] = bswapRecord{x, ^x}
	}
	for _, n := range []int{0, 1, 2, 17, 1024} {
		var wantLate, wantFields uint64
		for i := 0; i < n; i++ {
			wantLate += words[i]
			wantFields += bswapReference(records[i].field, 16) ^ records[i].payload
		}
		if got := bswapLateDispatch(words[:n]); got != wantLate {
			t.Fatalf("late dispatch length %d = %x, want %x", n, got, wantLate)
		}
		if got := bswapNetworkByteFields(records[:n]); got != wantFields {
			t.Fatalf("network fields length %d = %x, want %x", n, got, wantFields)
		}
		for _, initial := range []uint64{0, 1, 2, 3, 1 << 63, ^uint64(0)} {
			state, want := initial, uint64(0)
			for _, word := range words[:n] {
				state *= state
				if bswapReference(state, 64) == uint64(1)<<56 {
					want += word
				} else {
					want ^= word
				}
			}
			if got := bswapDynamicDispatch(words[:n], initial); got != want {
				t.Fatalf("dynamic dispatch length %d state %x = %x, want %x", n, initial, got, want)
			}
		}
	}
	if bswapLateDispatch(nil) != 0 || bswapNetworkByteFields(nil) != 0 {
		t.Fatal("nil input produced a nonzero checksum")
	}
	original := bswapRecord{0x1234, 0x23456789abcdef01}
	before := bswapNetworkByteFields([]bswapRecord{original})
	for _, changed := range []bswapRecord{
		{original.field ^ 1, original.payload},
		{original.field, original.payload ^ 1},
	} {
		want := bswapReference(changed.field, 16) ^ changed.payload
		if got := bswapNetworkByteFields([]bswapRecord{changed}); got != want || got == before {
			t.Fatalf("record field stopped affecting checksum: %+v -> %x, want %x", changed, got, want)
		}
	}
}

func TestScalarBswapEffects(t *testing.T) {
	calls := 0
	if got := bswapMaskedCall(func() uint64 {
		calls++
		return 0x0123456789abcdef
	}); got != 0 || calls != 1 {
		t.Fatalf("masked call result=%x calls=%d, want zero and one call", got, calls)
	}
	for _, tc := range []struct {
		name string
		call func() uint64
	}{
		{"nil_load", func() uint64 { return bswapMaskedLoad(nil) }},
		{"negative_index", func() uint64 { return bswapMaskedIndex([]uint64{1}, -1) }},
		{"past_end_index", func() uint64 { return bswapMaskedIndex([]uint64{1}, 1) }},
		{"division_by_zero", func() uint64 { return bswapMaskedDivision(1, 0) }},
		{"nil_call", func() uint64 { return bswapMaskedCall(nil) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("required panic was eliminated")
				}
			}()
			sink = tc.call()
		})
	}
}

func BenchmarkBswapLateDispatch(b *testing.B) {
	var words [1024]uint64
	for i := range words {
		words[i] = uint64(i+1) * 0x9e3779b97f4a7c15
	}
	b.SetBytes(int64(len(words)) * 8)
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += bswapLateDispatch(words[:])
	}
	sink = checksum
}

func BenchmarkBswapNetworkByteFields(b *testing.B) {
	var records [1024]bswapRecord
	for i := range records {
		x := uint64(i+1) * 0x9e3779b97f4a7c15
		records[i] = bswapRecord{x, ^x}
	}
	b.SetBytes(int64(len(records)) * 16)
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += bswapNetworkByteFields(records[:])
	}
	sink = checksum
}
