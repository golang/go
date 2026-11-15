// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonscan

import (
	"internal/cpu"
	"strconv"
	"testing"
	"unsafe"
)

const (
	maxLen       = 300
	maxAlign     = 64
	guardLen     = 64
	byteSweepLen = 80
	arenaLen     = guardLen + maxAlign + maxLen + 1 + guardLen
)

type class struct {
	name    string
	member  func(byte) bool
	bytes   func([]byte) int
	str     func(string) int
	ref     func([]byte) int
	clean   []byte
	special []byte
}

var (
	stringSpecialClass = splitBytes(&class{
		name:   "IndexStringSpecial",
		member: isStringSpecial,
		bytes:  IndexStringSpecial[[]byte],
		str:    IndexStringSpecial[string],
		ref:    indexStringSpecialRef[[]byte],
	})
	escapeHTMLClass = splitBytes(&class{
		name:   "IndexEscapeHTML",
		member: isEscapeHTML,
		bytes:  IndexEscapeHTML[[]byte],
		str:    IndexEscapeHTML[string],
		ref:    indexEscapeHTMLRef[[]byte],
	})
	escapeClass = splitBytes(&class{
		name:   "IndexEscape",
		member: isStringSpecial,
		bytes:  IndexEscape[[]byte],
		str:    IndexEscape[string],
		ref:    indexStringSpecialRef[[]byte],
	})
	classes = []*class{stringSpecialClass, escapeHTMLClass, escapeClass}
)

var probes = []byte{
	0x00, 0x01, '\n', 0x1f, 0x20, '!', '"', '#', '%', '&', '\'', ';', '<', '=', '>', '?',
	'[', '\\', ']', '~', 0x7f, 0x80, 0xc3, 0xfe, 0xff,
}

var allBytes = func() []byte {
	b := make([]byte, 256)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}()

func splitBytes(c *class) *class {
	for _, v := range allBytes {
		if c.member(v) {
			c.special = append(c.special, v)
		} else {
			c.clean = append(c.clean, v)
		}
	}
	return c
}

func forEachKernel(t *testing.T, f func(*testing.T)) {
	t.Run("default", f)
	if !cpu.X86.HasAVX2 {
		return
	}
	cpu.X86.HasAVX2 = false
	defer func() { cpu.X86.HasAVX2 = true }()
	t.Run("SSE2", f)
}

func check(t *testing.T, c *class, b []byte, want int) {
	ref := c.ref(b)
	got := c.bytes(b)
	gotString := c.str(unsafe.String(unsafe.SliceData(b), len(b)))
	if ref == want && got == want && gotString == want {
		return
	}
	t.Helper()
	t.Fatalf("%s: len %d, address mod 64 %d: []byte %d, string %d, reference %d, want %d\ninput %q",
		c.name, len(b), uintptr(unsafe.Pointer(unsafe.SliceData(b)))%maxAlign, got, gotString, ref, want, b)
}

func fill(dst, pattern []byte, phase int) {
	for i := range dst {
		dst[i] = pattern[(i+phase)%len(pattern)]
	}
}

func place(c *class, arena []byte, off, n int) []byte {
	fill(arena, c.special, n)
	start := guardLen + off
	w := arena[start : start+n : start+n]
	fill(w, c.clean, off+n)
	arena[start+n] = c.clean[0]
	return w
}

func expectAt(c *class, v byte, i, n int) int {
	if c.member(v) {
		return i
	}
	return n
}

func probeEveryPosition(t *testing.T, c *class, w []byte, v byte) {
	for i := range w {
		keep := w[i]
		w[i] = v
		check(t, c, w, expectAt(c, v, i, len(w)))
		w[i] = keep
	}
}

func sweepPositions(t *testing.T, c *class, values []byte, maxN int) {
	arena := make([]byte, arenaLen)
	for n := 1; n <= maxN; n++ {
		w := place(c, arena, n%maxAlign, n)
		for _, v := range values {
			probeEveryPosition(t, c, w, v)
		}
	}
}

func checkCleanWindows(t *testing.T, c *class) {
	arena := make([]byte, arenaLen)
	for off := range maxAlign {
		for n := range maxLen + 1 {
			check(t, c, place(c, arena, off, n), n)
		}
	}
}

func atBackingEnd(c *class, n int) []byte {
	b := make([]byte, guardLen+n)
	fill(b[:guardLen], c.special, n)
	fill(b[guardLen:], c.clean, n)
	return b[guardLen:]
}

func atBackingStart(c *class, n int) []byte {
	b := make([]byte, n+guardLen)
	fill(b[:n], c.clean, n)
	fill(b[n:], c.special, n)
	return b[:n:n]
}

func checkBackingBounds(t *testing.T, c *class) {
	for n := range maxLen + 1 {
		check(t, c, atBackingEnd(c, n), n)
		check(t, c, atBackingStart(c, n), n)
	}
}

func TestCleanEveryLengthAndAlignment(t *testing.T) {
	forEachKernel(t, func(t *testing.T) {
		for _, c := range classes {
			checkCleanWindows(t, c)
		}
	})
}

func TestProbeEveryLengthAndPosition(t *testing.T) {
	forEachKernel(t, func(t *testing.T) {
		for _, c := range classes {
			sweepPositions(t, c, probes, maxLen)
		}
	})
}

func TestEveryByteEveryPosition(t *testing.T) {
	forEachKernel(t, func(t *testing.T) {
		for _, c := range classes {
			sweepPositions(t, c, allBytes, byteSweepLen)
		}
	})
}

func TestInputAtBackingArrayBounds(t *testing.T) {
	forEachKernel(t, func(t *testing.T) {
		for _, c := range classes {
			checkBackingBounds(t, c)
		}
	})
}

var fuzzSeeds = []string{
	"",
	"plain ascii text without escapes",
	`a "quoted" word in a sentence`,
	`back\slash in the middle of text`,
	"<script>alert(1)</script>&amp;",
	"control \x00\x01\x1f bytes and DEL \x7f",
	"high bytes \x80\xc3\xa9\xe2\x80\xa8\xff",
	"0123456789abcdef\"",
	"0123456789abcdef0123456789abcde\\",
	"0123456789abcdef0123456789abcdef<",
}

func fuzzClass(f *testing.F, c *class) {
	for _, s := range fuzzSeeds {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		for off := range min(len(b), maxAlign) + 1 {
			check(t, c, b[off:], c.ref(b[off:]))
		}
	})
}

func FuzzIndexStringSpecial(f *testing.F) { fuzzClass(f, stringSpecialClass) }

func FuzzIndexEscapeHTML(f *testing.F) { fuzzClass(f, escapeHTMLClass) }

func FuzzIndexEscape(f *testing.F) { fuzzClass(f, escapeClass) }

var benchLens = []int{8, 16, 64, 1024, 65536}

func benchScan(b *testing.B, scan func([]byte) int, in []byte) {
	b.SetBytes(int64(len(in)))
	for b.Loop() {
		if scan(in) != len(in) {
			b.Fatal("clean input reported a special byte")
		}
	}
}

func benchClass(b *testing.B, c *class) {
	for _, n := range benchLens {
		in := make([]byte, n)
		fill(in, c.clean, 0)
		b.Run("kernel/"+strconv.Itoa(n), func(b *testing.B) { benchScan(b, c.bytes, in) })
		b.Run("reference/"+strconv.Itoa(n), func(b *testing.B) { benchScan(b, c.ref, in) })
	}
}

func BenchmarkIndexStringSpecial(b *testing.B) { benchClass(b, stringSpecialClass) }

func BenchmarkIndexEscapeHTML(b *testing.B) { benchClass(b, escapeHTMLClass) }

func BenchmarkIndexEscape(b *testing.B) { benchClass(b, escapeClass) }
