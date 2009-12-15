// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf8_test

import (
	"bytes"
	"strings"
	"testing"
	. "utf8"
)

type Utf8Map struct {
	rune int
	str  string
}

var utf8map = []Utf8Map{
	Utf8Map{0x0000, "\x00"},
	Utf8Map{0x0001, "\x01"},
	Utf8Map{0x007e, "\x7e"},
	Utf8Map{0x007f, "\x7f"},
	Utf8Map{0x0080, "\xc2\x80"},
	Utf8Map{0x0081, "\xc2\x81"},
	Utf8Map{0x00bf, "\xc2\xbf"},
	Utf8Map{0x00c0, "\xc3\x80"},
	Utf8Map{0x00c1, "\xc3\x81"},
	Utf8Map{0x00c8, "\xc3\x88"},
	Utf8Map{0x00d0, "\xc3\x90"},
	Utf8Map{0x00e0, "\xc3\xa0"},
	Utf8Map{0x00f0, "\xc3\xb0"},
	Utf8Map{0x00f8, "\xc3\xb8"},
	Utf8Map{0x00ff, "\xc3\xbf"},
	Utf8Map{0x0100, "\xc4\x80"},
	Utf8Map{0x07ff, "\xdf\xbf"},
	Utf8Map{0x0800, "\xe0\xa0\x80"},
	Utf8Map{0x0801, "\xe0\xa0\x81"},
	Utf8Map{0xfffe, "\xef\xbf\xbe"},
	Utf8Map{0xffff, "\xef\xbf\xbf"},
	Utf8Map{0x10000, "\xf0\x90\x80\x80"},
	Utf8Map{0x10001, "\xf0\x90\x80\x81"},
	Utf8Map{0x10fffe, "\xf4\x8f\xbf\xbe"},
	Utf8Map{0x10ffff, "\xf4\x8f\xbf\xbf"},
	Utf8Map{0xFFFD, "\xef\xbf\xbd"},
}

// strings.Bytes with one extra byte at end
func makeBytes(s string) []byte {
	s += "\x00"
	b := strings.Bytes(s)
	return b[0 : len(s)-1]
}

func TestFullRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i]
		b := makeBytes(m.str)
		if !FullRune(b) {
			t.Errorf("FullRune(%q) (rune %04x) = false, want true", b, m.rune)
		}
		s := m.str
		if !FullRuneInString(s) {
			t.Errorf("FullRuneInString(%q) (rune %04x) = false, want true", s, m.rune)
		}
		b1 := b[0 : len(b)-1]
		if FullRune(b1) {
			t.Errorf("FullRune(%q) = true, want false", b1)
		}
		s1 := string(b1)
		if FullRuneInString(s1) {
			t.Errorf("FullRune(%q) = true, want false", s1)
		}
	}
}

func TestEncodeRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i]
		b := makeBytes(m.str)
		var buf [10]byte
		n := EncodeRune(m.rune, &buf)
		b1 := buf[0:n]
		if !bytes.Equal(b, b1) {
			t.Errorf("EncodeRune(%#04x) = %q want %q", m.rune, b1, b)
		}
	}
}

func TestDecodeRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i]
		b := makeBytes(m.str)
		rune, size := DecodeRune(b)
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", b, rune, size, m.rune, len(b))
		}
		s := m.str
		rune, size = DecodeRuneInString(s)
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", s, rune, size, m.rune, len(b))
		}

		// there's an extra byte that bytes left behind - make sure trailing byte works
		rune, size = DecodeRune(b[0:cap(b)])
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", b, rune, size, m.rune, len(b))
		}
		s = m.str + "\x00"
		rune, size = DecodeRuneInString(s)
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRuneInString(%q) = %#04x, %d want %#04x, %d", s, rune, size, m.rune, len(b))
		}

		// make sure missing bytes fail
		wantsize := 1
		if wantsize >= len(b) {
			wantsize = 0
		}
		rune, size = DecodeRune(b[0 : len(b)-1])
		if rune != RuneError || size != wantsize {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", b[0:len(b)-1], rune, size, RuneError, wantsize)
		}
		s = m.str[0 : len(m.str)-1]
		rune, size = DecodeRuneInString(s)
		if rune != RuneError || size != wantsize {
			t.Errorf("DecodeRuneInString(%q) = %#04x, %d want %#04x, %d", s, rune, size, RuneError, wantsize)
		}

		// make sure bad sequences fail
		if len(b) == 1 {
			b[0] = 0x80
		} else {
			b[len(b)-1] = 0x7F
		}
		rune, size = DecodeRune(b)
		if rune != RuneError || size != 1 {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", b, rune, size, RuneError, 1)
		}
		s = string(b)
		rune, size = DecodeRune(b)
		if rune != RuneError || size != 1 {
			t.Errorf("DecodeRuneInString(%q) = %#04x, %d want %#04x, %d", s, rune, size, RuneError, 1)
		}
	}
}

// Check that negative runes encode as U+FFFD.
func TestNegativeRune(t *testing.T) {
	errorbuf := make([]byte, UTFMax)
	errorbuf = errorbuf[0:EncodeRune(RuneError, errorbuf)]
	buf := make([]byte, UTFMax)
	buf = buf[0:EncodeRune(-1, buf)]
	if !bytes.Equal(buf, errorbuf) {
		t.Errorf("incorrect encoding [% x] for -1; expected [% x]", buf, errorbuf)
	}
}

type RuneCountTest struct {
	in  string
	out int
}

var runecounttests = []RuneCountTest{
	RuneCountTest{"abcd", 4},
	RuneCountTest{"☺☻☹", 3},
	RuneCountTest{"1,2,3,4", 7},
	RuneCountTest{"\xe2\x00", 2},
}

func TestRuneCount(t *testing.T) {
	for i := 0; i < len(runecounttests); i++ {
		tt := runecounttests[i]
		if out := RuneCountInString(tt.in); out != tt.out {
			t.Errorf("RuneCountInString(%q) = %d, want %d", tt.in, out, tt.out)
		}
		if out := RuneCount(makeBytes(tt.in)); out != tt.out {
			t.Errorf("RuneCount(%q) = %d, want %d", tt.in, out, tt.out)
		}
	}
}

func BenchmarkRuneCountTenASCIIChars(b *testing.B) {
	for i := 0; i < b.N; i++ {
		RuneCountInString("0123456789")
	}
}

func BenchmarkRuneCountTenJapaneseChars(b *testing.B) {
	for i := 0; i < b.N; i++ {
		RuneCountInString("日本語日本語日本語日")
	}
}

func BenchmarkEncodeASCIIRune(b *testing.B) {
	buf := make([]byte, UTFMax)
	for i := 0; i < b.N; i++ {
		EncodeRune('a', buf)
	}
}

func BenchmarkEncodeJapaneseRune(b *testing.B) {
	buf := make([]byte, UTFMax)
	for i := 0; i < b.N; i++ {
		EncodeRune('本', buf)
	}
}

func BenchmarkDecodeASCIIRune(b *testing.B) {
	a := []byte{'a'}
	for i := 0; i < b.N; i++ {
		DecodeRune(a)
	}
}

func BenchmarkDecodeJapaneseRune(b *testing.B) {
	nihon := strings.Bytes("本")
	for i := 0; i < b.N; i++ {
		DecodeRune(nihon)
	}
}
