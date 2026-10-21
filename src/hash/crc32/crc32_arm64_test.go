// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"fmt"
	"internal/cpu"
	"math/rand"
	"testing"
)

// SHA3 is not required, so the tests also run on Neoverse N1.
func hasPMULLCRC32() bool {
	return cpu.ARM64.HasPMULL && cpu.ARM64.HasCRC32
}

func TestCastagnoliPMULL(t *testing.T) {
	if !hasPMULLCRC32() {
		t.Skip("PMULL and CRC32 not available")
	}
	slicingTable := slicingMakeTable(Castagnoli)
	slicing := func(crc uint32, b []byte) uint32 {
		return slicingUpdate(crc, slicingTable, b)
	}
	pmull := func(crc uint32, b []byte) uint32 {
		return ^castagnoliUpdatePMULL(^crc, b)
	}
	testCrossCheck(t, pmull, slicing)

	lengths := make([]int, 0, 4100)
	for n := 0; n <= 4096; n++ {
		lengths = append(lengths, n)
	}
	lengths = append(lengths, 1<<15, 1<<20, 1<<20+7)
	buf := make([]byte, lengths[len(lengths)-1])
	rand.Read(buf)
	for _, n := range lengths {
		crcInit := uint32(rand.Int63())
		want := slicing(crcInit, buf[:n])
		if got := pmull(crcInit, buf[:n]); got != want {
			t.Errorf("mismatch: 0x%x vs 0x%x (buffer length %d)", got, want, n)
		}
	}
}

// Every cache-line offset, around each change of loop count.
func TestCastagnoliPMULLAlignment(t *testing.T) {
	if !hasPMULLCRC32() {
		t.Skip("PMULL and CRC32 not available")
	}
	slicingTable := slicingMakeTable(Castagnoli)
	buf := make([]byte, 64+2304)
	rand.Read(buf)
	for off := 0; off < 64; off++ {
		for base := 0; base <= 2048; base += 128 {
			for d := -17; d <= 17; d++ {
				n := base + d
				if n < 0 {
					continue
				}
				p := buf[off : off+n]
				for _, init := range []uint32{0, 0xffffffff, uint32(rand.Int63())} {
					want := slicingUpdate(init, slicingTable, p)
					if got := ^castagnoliUpdatePMULL(^init, p); got != want {
						t.Fatalf("offset %d length %d crc %#x: got %#x, want %#x", off, n, init, got, want)
					}
				}
			}
		}
	}
}

// Splits cross pmullMinLength in both directions.
func TestCastagnoliSplitWrites(t *testing.T) {
	tab := MakeTable(Castagnoli)
	buf := make([]byte, 3*pmullMinLength+5)
	rand.Read(buf)
	want := Checksum(buf, tab)
	for i := 0; i <= len(buf); i++ {
		h := New(tab)
		h.Write(buf[:i])
		h.Write(buf[i:])
		if got := h.Sum32(); got != want {
			t.Fatalf("split at %d: got %#x, want %#x", i, got, want)
		}
	}
}

var benchSink uint32

// BenchmarkCastagnoliARM64 bypasses the dispatch to compare both paths.
func BenchmarkCastagnoliARM64(b *testing.B) {
	if !hasPMULLCRC32() {
		b.Skip("PMULL and CRC32 not available")
	}
	sizes := []int{128, 192, 256, 384, 512, 1 << 10, 4 << 10, 32 << 10, 1 << 20}
	impls := []struct {
		name string
		fn   func(uint32, []byte) uint32
	}{
		{"scalar", castagnoliUpdate},
		{"pmull", castagnoliUpdatePMULL},
	}
	for _, size := range sizes {
		data := make([]byte, size)
		for i := range data {
			data[i] = byte(i)
		}
		for _, impl := range impls {
			b.Run(fmt.Sprintf("size=%d/independent/%s", size, impl.name), func(b *testing.B) {
				b.SetBytes(int64(size))
				for range b.N {
					benchSink = impl.fn(0, data)
				}
			})
			b.Run(fmt.Sprintf("size=%d/chained/%s", size, impl.name), func(b *testing.B) {
				b.SetBytes(int64(size))
				crc := uint32(0)
				for range b.N {
					crc = impl.fn(crc, data)
				}
				benchSink = crc
			})
		}
	}
}
