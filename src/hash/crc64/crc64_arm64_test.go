// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc64

import (
	"fmt"
	"internal/cpu"
	"math/rand"
	"testing"
)

// bytewiseUpdate is the reference: one table lookup per byte.
func bytewiseUpdate(crc uint64, tab *Table, p []byte) uint64 {
	crc = ^crc
	for _, v := range p {
		crc = tab[byte(crc)^v] ^ (crc >> 8)
	}
	return ^crc
}

func foldTestTables() map[string]*Table {
	ecmaCopy := *MakeTable(ECMA) // equal contents, different pointer
	return map[string]*Table{
		"ISO":      MakeTable(ISO),
		"ECMA":     MakeTable(ECMA),
		"ECMAcopy": &ecmaCopy,
		"custom":   MakeTable(0x777),
	}
}

// TestPMULLFoldPath checks which tables take the fold, since checksums
// match either way.
func TestPMULLFoldPath(t *testing.T) {
	if !cpu.ARM64.HasPMULL {
		t.Skip("PMULL not available.")
	}
	p := make([]byte, 1000)
	for name, tab := range foldTestTables() {
		_, rest := archFold(0, tab, p)
		folded := len(rest) < 8
		if want := name == "ISO" || name == "ECMA"; folded != want {
			t.Errorf("%s: folded=%v, want %v", name, folded, want)
		}
	}
	for _, tab := range []*Table{MakeTable(ISO), MakeTable(ECMA)} {
		for _, n := range []int{crc64FoldMinLength - 1, crc64FoldMinLength} {
			_, rest := archFold(0, tab, p[:n])
			if folded, want := len(rest) < n, n >= crc64FoldMinLength; folded != want {
				t.Errorf("len=%d: folded=%v, want %v", n, folded, want)
			}
		}
	}
	if _, rest := foldPMULL(0, MakeTable(0x777), p); len(rest) != len(p) {
		t.Errorf("foldPMULL folded a custom table")
	}
	defer func(v bool) { usePMULL = v }(usePMULL)
	usePMULL = false
	if _, rest := archFold(0, MakeTable(ECMA), p); len(rest) != len(p) {
		t.Errorf("usePMULL=false: folded %d bytes", len(p)-len(rest))
	}
}

// TestPMULLFold runs with PMULL (if available) and with it forced off.
func TestPMULLFold(t *testing.T) {
	t.Run("pmull", func(t *testing.T) {
		if !cpu.ARM64.HasPMULL {
			t.Skip("PMULL not available.")
		}
		testFold(t)
	})
	t.Run("nopmull", func(t *testing.T) {
		defer func(v bool) { usePMULL = v }(usePMULL)
		usePMULL = false
		testFold(t)
	})
}

func testFold(t *testing.T) {
	buf := make([]byte, 1<<20+64)
	rand.Read(buf)

	lengths := make([]int, 0, 4100)
	for n := 0; n <= 4096; n++ {
		lengths = append(lengths, n)
	}
	lengths = append(lengths, 1<<15, 1<<20, 1<<20+7, 100000, 65537)

	for name, tab := range foldTestTables() {
		for _, n := range lengths {
			want := bytewiseUpdate(0, tab, buf[:n])
			if got := Checksum(buf[:n], tab); got != want {
				t.Fatalf("%s len=%d: got %#x want %#x", name, n, got, want)
			}
		}
		// Unaligned starts.
		for off := 1; off < 16; off++ {
			for _, n := range []int{15, 16, 17, 31, 32, 47, 63, 64, 127, 128, 129, 191, 192, 255, 256, 1000, 2048, 4096, 65537} {
				p := buf[off : off+n]
				if got, want := Checksum(p, tab), bytewiseUpdate(0, tab, p); got != want {
					t.Fatalf("%s off=%d len=%d: got %#x want %#x", name, off, n, got, want)
				}
			}
		}
		// Non-initial crc carried into the fold.
		for _, n := range []int{16, 40, 128, 200, 1000, 4096, 1 << 16} {
			p := buf[:n]
			want := bytewiseUpdate(0, tab, p)
			for _, split := range []int{1, 8, 63, 64, 128, 129, n / 2, n - 128} {
				if split < 0 || split > n {
					continue
				}
				if got := Update(Update(0, tab, p[:split]), tab, p[split:]); got != want {
					t.Fatalf("%s Update len=%d split=%d: got %#x want %#x", name, n, split, got, want)
				}
				h := New(tab)
				h.Write(p[:split])
				h.Write(p[split:])
				if got := h.Sum64(); got != want {
					t.Fatalf("%s Write len=%d split=%d: got %#x want %#x", name, n, split, got, want)
				}
			}
		}
	}
}

func FuzzPMULLFold(f *testing.F) {
	f.Add(make([]byte, 300), uint64(0), 7)
	f.Fuzz(func(t *testing.T, p []byte, crc uint64, split int) {
		if split < 0 || split > len(p) {
			split = len(p) / 2
		}
		for name, tab := range foldTestTables() {
			want := bytewiseUpdate(crc, tab, p)
			if got := Update(Update(crc, tab, p[:split]), tab, p[split:]); got != want {
				t.Fatalf("%s len=%d split=%d: got %#x want %#x", name, len(p), split, got, want)
			}
		}
	})
}

func BenchmarkCRC64Fold(b *testing.B) {
	sizes := []int{16, 32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 768,
		1 << 10, 2 << 10, 4 << 10, 8 << 10, 16 << 10, 32 << 10,
		64 << 10, 256 << 10, 1 << 20}
	tab := MakeTable(ECMA)
	for _, size := range sizes {
		data := make([]byte, size)
		for i := range data {
			data[i] = byte(i)
		}
		for _, pmull := range []bool{false, true} {
			name := fmt.Sprintf("%d/scalar", size)
			if pmull {
				name = fmt.Sprintf("%d/pmull", size)
			}
			b.Run(name, func(b *testing.B) {
				if pmull && !cpu.ARM64.HasPMULL {
					b.Skip("PMULL not available.")
				}
				defer func(v bool) { usePMULL = v }(usePMULL)
				usePMULL = pmull
				b.SetBytes(int64(size))
				for range b.N {
					Checksum(data, tab)
				}
			})
		}
	}
}
