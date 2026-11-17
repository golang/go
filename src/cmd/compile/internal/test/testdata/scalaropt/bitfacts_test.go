// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalaropt

import (
	"math/bits"
	"testing"
)

type bitCountRecord struct {
	word, payload uint64
}

func consumeAlignedCount(word, payload uint64) uint64 {
	return (word >> uint(bits.TrailingZeros64(word))) ^ payload
}

func consumeLengthCount(word, payload uint64) uint64 {
	return word ^ (payload >> uint(bits.Len64(word)))
}

func consumePopulationClass(word, payload uint64) uint64 {
	return word ^ (payload >> uint(bits.OnesCount64(word)>>4))
}

// The low byte encodes a fixed alignment class, while the upper address bits
// remain dynamic and contribute to the checksum.
//
//go:noinline
func alignedBitCounts(records []bitCountRecord) uint64 {
	var checksum uint64
	for _, r := range records {
		word := (r.word &^ 255) + 32
		checksum += consumeAlignedCount(word, r.payload)
	}
	return checksum
}

// The marker above the 16-bit key sets a fixed length after wire rotation.
// The key and the payload still contribute to every record's result.
//
//go:noinline
func headerBitLengths(records []bitCountRecord) uint64 {
	var checksum uint64
	for _, r := range records {
		word := bits.RotateLeft64(uint64(uint16(r.word))|1<<16, 24)
		checksum += consumeLengthCount(word, r.payload)
	}
	return checksum
}

// Sixteen fixed one bits and fifteen optional bits put the population in
// [16, 31]. Only its upper class bits select how much payload is consumed.
//
//go:noinline
func sparsePopulationClasses(records []bitCountRecord) uint64 {
	var checksum uint64
	for _, r := range records {
		word := (r.word & 0x15555555) | 0xaaaaaaaa00000000
		checksum += consumePopulationClass(word, r.payload)
	}
	return checksum
}

func TestScalarBitFactsProcessing(t *testing.T) {
	var extremes []bitCountRecord
	for _, word := range []uint64{0, 1, 31, 32, 255, 256, 1 << 63, ^uint64(0), 0xaaaaaaaaaaaaaaaa, 0x5555555555555555} {
		for _, payload := range []uint64{0, 1, 2, 1 << 41, 1 << 63, ^uint64(0)} {
			extremes = append(extremes, bitCountRecord{word, payload})
		}
	}
	for _, tc := range []struct {
		name    string
		records []bitCountRecord
	}{
		{"nil", nil},
		{"empty", []bitCountRecord{}},
		{"singleton", []bitCountRecord{{0x123456789abcdef0, 0xfedcba9876543210}}},
		{"extremes", extremes},
		{"mixed", makeBitCountRecords()},
		{"wrapping-checksum", []bitCountRecord{{0, ^uint64(0)}, {256, ^uint64(0)}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var aligned, length, population uint64
			for _, r := range tc.records {
				aligned += ((r.word>>8)<<3 | 1) ^ r.payload
				wire := (uint64(uint16(r.word)) | 1<<16) << 24
				length += wire ^ (r.payload >> 41)
				sparse := (r.word & 0x15555555) | 0xaaaaaaaa00000000
				population += sparse ^ (r.payload >> 1)
			}
			if got := alignedBitCounts(tc.records); got != aligned {
				t.Fatalf("aligned checksum = %#x, want %#x", got, aligned)
			}
			if got := headerBitLengths(tc.records); got != length {
				t.Fatalf("header checksum = %#x, want %#x", got, length)
			}
			if got := sparsePopulationClasses(tc.records); got != population {
				t.Fatalf("population checksum = %#x, want %#x", got, population)
			}
		})
	}
	for _, tc := range []struct {
		name                string
		process             func([]bitCountRecord) uint64
		wordBit, payloadBit uint64
	}{
		{"aligned", alignedBitCounts, 256, 1},
		{"header", headerBitLengths, 1, 1 << 41},
		{"population", sparsePopulationClasses, 1, 2},
	} {
		original := tc.process([]bitCountRecord{{0, 0}})
		for _, changed := range []bitCountRecord{{tc.wordBit, 0}, {0, tc.payloadBit}} {
			if got := tc.process([]bitCountRecord{changed}); got == original {
				t.Fatalf("%s ignored dynamic record field: %+v", tc.name, changed)
			}
		}
	}
}

// No math/bits calls appear in the generic-control oracle.
func referenceBitCounts(word uint64) (length, trailing, population uint) {
	trailing = 64
	for bit := uint(0); bit < 64; bit++ {
		if word&(uint64(1)<<bit) != 0 {
			length = bit + 1
			population++
			if trailing == 64 {
				trailing = bit
			}
		}
	}
	return
}

func TestScalarBitFactsGenericControls(t *testing.T) {
	values := []uint64{0, 1, 2, 31, 32, 255, 256, 1 << 63, ^uint64(0), 0xaaaaaaaaaaaaaaaa, 0x5555555555555555}
	for i := uint64(0); i < 4096; i++ {
		values = append(values, i*0x9e3779b97f4a7c15)
	}
	for _, word := range values {
		length, trailing, population := referenceBitCounts(word)
		payload := ^word
		if got, want := consumeAlignedCount(word, payload), (word>>trailing)^payload; got != want {
			t.Fatalf("generic trailing count for %#x = %#x, want %#x", word, got, want)
		}
		if got, want := consumeLengthCount(word, payload), word^(payload>>length); got != want {
			t.Fatalf("generic bit length for %#x = %#x, want %#x", word, got, want)
		}
		if got, want := consumePopulationClass(word, payload), word^(payload>>(population>>4)); got != want {
			t.Fatalf("generic population class for %#x = %#x, want %#x", word, got, want)
		}
	}
}

//go:noinline
func bitFactsLoad(p *uint16) int {
	return bits.Len64(bits.RotateLeft64(uint64(*p)|1<<16, 24))
}

//go:noinline
func bitFactsIndex(xs []uint64, index int) int {
	return bits.TrailingZeros64((xs[index] &^ 255) + 32)
}

//go:noinline
func bitFactsDivide(x, divisor uint64) int {
	return bits.TrailingZeros64(((x / divisor) &^ 255) + 32)
}

//go:noinline
func countedBitFactsInput(calls *int, word uint64) uint64 {
	*calls += 1
	return word
}

func TestScalarBitFactsPanicsAndEffects(t *testing.T) {
	for _, tc := range []struct {
		name string
		call func() int
	}{
		{"nil-load", func() int { return bitFactsLoad(nil) }},
		{"negative-index", func() int { return bitFactsIndex([]uint64{1}, -1) }},
		{"past-end-index", func() int { return bitFactsIndex([]uint64{1}, 1) }},
		{"division-by-zero", func() int { return bitFactsDivide(1, 0) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("required panic was eliminated")
				}
			}()
			sink = uint64(tc.call())
		})
	}
	calls := 0
	word := (countedBitFactsInput(&calls, 7) & 65535) | 1<<16
	if got := bits.Len64(bits.RotateLeft64(word, 24)); got != 41 || calls != 1 {
		t.Fatalf("counted input: length=%d calls=%d, want 41 and 1", got, calls)
	}
}

func makeBitCountRecords() []bitCountRecord {
	records := make([]bitCountRecord, 1024)
	for i := range records {
		x := uint64(i+1) * 0x9e3779b97f4a7c15
		records[i] = bitCountRecord{x, ^x}
	}
	return records
}

func BenchmarkAlignedBitCounts(b *testing.B) {
	records := makeBitCountRecords()
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += alignedBitCounts(records)
	}
	sink = checksum
}

func BenchmarkHeaderBitLengths(b *testing.B) {
	records := makeBitCountRecords()
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += headerBitLengths(records)
	}
	sink = checksum
}

func BenchmarkSparsePopulationClasses(b *testing.B) {
	records := makeBitCountRecords()
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += sparsePopulationClasses(records)
	}
	sink = checksum
}
