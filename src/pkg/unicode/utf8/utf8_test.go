// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf8_test

import (
	"bytes"
	"testing"
	"unicode"
	. "unicode/utf8"
)

// Validate the constants redefined from unicode.
func init() {
	if MaxRune != unicode.MaxRune {
		panic("utf8.MaxRune is wrong")
	}
	if RuneError != unicode.ReplacementChar {
		panic("utf8.RuneError is wrong")
	}
}

// Validate the constants redefined from unicode.
func TestConstants(t *testing.T) {
	if MaxRune != unicode.MaxRune {
		t.Errorf("utf8.MaxRune is wrong: %x should be %x", MaxRune, unicode.MaxRune)
	}
	if RuneError != unicode.ReplacementChar {
		t.Errorf("utf8.RuneError is wrong: %x should be %x", RuneError, unicode.ReplacementChar)
	}
}

type Utf8Map struct {
	r   rune
	str string
}

var utf8map = []Utf8Map{
	{0x0000, "\x00"},
	{0x0001, "\x01"},
	{0x007e, "\x7e"},
	{0x007f, "\x7f"},
	{0x0080, "\xc2\x80"},
	{0x0081, "\xc2\x81"},
	{0x00bf, "\xc2\xbf"},
	{0x00c0, "\xc3\x80"},
	{0x00c1, "\xc3\x81"},
	{0x00c8, "\xc3\x88"},
	{0x00d0, "\xc3\x90"},
	{0x00e0, "\xc3\xa0"},
	{0x00f0, "\xc3\xb0"},
	{0x00f8, "\xc3\xb8"},
	{0x00ff, "\xc3\xbf"},
	{0x0100, "\xc4\x80"},
	{0x07ff, "\xdf\xbf"},
	{0x0800, "\xe0\xa0\x80"},
	{0x0801, "\xe0\xa0\x81"},
	{0xd7ff, "\xed\x9f\xbf"}, // last code point before surrogate half.
	{0xe000, "\xee\x80\x80"}, // first code point after surrogate half.
	{0xfffe, "\xef\xbf\xbe"},
	{0xffff, "\xef\xbf\xbf"},
	{0x10000, "\xf0\x90\x80\x80"},
	{0x10001, "\xf0\x90\x80\x81"},
	{0x10fffe, "\xf4\x8f\xbf\xbe"},
	{0x10ffff, "\xf4\x8f\xbf\xbf"},
	{0xFFFD, "\xef\xbf\xbd"},
}

var surrogateMap = []Utf8Map{
	{0xd800, "\xed\xa0\x80"}, // surrogate min decodes to (RuneError, 1)
	{0xdfff, "\xed\xbf\xbf"}, // surrogate max decodes to (RuneError, 1)
}

var testStrings = []string{
	"",
	"abcd",
	"☺☻☹",
	"日a本b語ç日ð本Ê語þ日¥本¼語i日©",
	"日a本b語ç日ð本Ê語þ日¥本¼語i日©日a本b語ç日ð本Ê語þ日¥本¼語i日©日a本b語ç日ð本Ê語þ日¥本¼語i日©",
	"\x80\x80\x80\x80",
}

func TestFullRune(t *testing.T) {
	for _, m := range utf8map {
		b := []byte(m.str)
		if !FullRune(b) {
			t.Errorf("FullRune(%q) (%U) = false, want true", b, m.r)
		}
		s := m.str
		if !FullRuneInString(s) {
			t.Errorf("FullRuneInString(%q) (%U) = false, want true", s, m.r)
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
	for _, m := range utf8map {
		b := []byte(m.str)
		var buf [10]byte
		n := EncodeRune(buf[0:], m.r)
		b1 := buf[0:n]
		if !bytes.Equal(b, b1) {
			t.Errorf("EncodeRune(%#04x) = %q want %q", m.r, b1, b)
		}
	}
}

func TestDecodeRune(t *testing.T) {
	for _, m := range utf8map {
		b := []byte(m.str)
		r, size := DecodeRune(b)
		if r != m.r || size != len(b) {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", b, r, size, m.r, len(b))
		}
		s := m.str
		r, size = DecodeRuneInString(s)
		if r != m.r || size != len(b) {
			t.Errorf("DecodeRuneInString(%q) = %#04x, %d want %#04x, %d", s, r, size, m.r, len(b))
		}

		// there's an extra byte that bytes left behind - make sure trailing byte works
		r, size = DecodeRune(b[0:cap(b)])
		if r != m.r || size != len(b) {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", b, r, size, m.r, len(b))
		}
		s = m.str + "\x00"
		r, size = DecodeRuneInString(s)
		if r != m.r || size != len(b) {
			t.Errorf("DecodeRuneInString(%q) = %#04x, %d want %#04x, %d", s, r, size, m.r, len(b))
		}

		// make sure missing bytes fail
		wantsize := 1
		if wantsize >= len(b) {
			wantsize = 0
		}
		r, size = DecodeRune(b[0 : len(b)-1])
		if r != RuneError || size != wantsize {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", b[0:len(b)-1], r, size, RuneError, wantsize)
		}
		s = m.str[0 : len(m.str)-1]
		r, size = DecodeRuneInString(s)
		if r != RuneError || size != wantsize {
			t.Errorf("DecodeRuneInString(%q) = %#04x, %d want %#04x, %d", s, r, size, RuneError, wantsize)
		}

		// make sure bad sequences fail
		if len(b) == 1 {
			b[0] = 0x80
		} else {
			b[len(b)-1] = 0x7F
		}
		r, size = DecodeRune(b)
		if r != RuneError || size != 1 {
			t.Errorf("DecodeRune(%q) = %#04x, %d want %#04x, %d", b, r, size, RuneError, 1)
		}
		s = string(b)
		r, size = DecodeRuneInString(s)
		if r != RuneError || size != 1 {
			t.Errorf("DecodeRuneInString(%q) = %#04x, %d want %#04x, %d", s, r, size, RuneError, 1)
		}

	}
}

func TestDecodeSurrogateRune(t *testing.T) {
	for _, m := range surrogateMap {
		b := []byte(m.str)
		r, size := DecodeRune(b)
		if r != RuneError || size != 1 {
			t.Errorf("DecodeRune(%q) = %x, %d want %x, %d", b, r, size, RuneError, 1)
		}
		s := m.str
		r, size = DecodeRuneInString(s)
		if r != RuneError || size != 1 {
			t.Errorf("DecodeRuneInString(%q) = %x, %d want %x, %d", b, r, size, RuneError, 1)
		}
	}
}

// Check that DecodeRune and DecodeLastRune correspond to
// the equivalent range loop.
func TestSequencing(t *testing.T) {
	for _, ts := range testStrings {
		for _, m := range utf8map {
			for _, s := range []string{ts + m.str, m.str + ts, ts + m.str + ts} {
				testSequence(t, s)
			}
		}
	}
}

// Check that a range loop and a []int conversion visit the same runes.
// Not really a test of this package, but the assumption is used here and
// it's good to verify
func TestIntConversion(t *testing.T) {
	for _, ts := range testStrings {
		runes := []rune(ts)
		if RuneCountInString(ts) != len(runes) {
			t.Errorf("%q: expected %d runes; got %d", ts, len(runes), RuneCountInString(ts))
			break
		}
		i := 0
		for _, r := range ts {
			if r != runes[i] {
				t.Errorf("%q[%d]: expected %c (%U); got %c (%U)", ts, i, runes[i], runes[i], r, r)
			}
			i++
		}
	}
}

func testSequence(t *testing.T, s string) {
	type info struct {
		index int
		r     rune
	}
	index := make([]info, len(s))
	b := []byte(s)
	si := 0
	j := 0
	for i, r := range s {
		if si != i {
			t.Errorf("Sequence(%q) mismatched index %d, want %d", s, si, i)
			return
		}
		index[j] = info{i, r}
		j++
		r1, size1 := DecodeRune(b[i:])
		if r != r1 {
			t.Errorf("DecodeRune(%q) = %#04x, want %#04x", s[i:], r1, r)
			return
		}
		r2, size2 := DecodeRuneInString(s[i:])
		if r != r2 {
			t.Errorf("DecodeRuneInString(%q) = %#04x, want %#04x", s[i:], r2, r)
			return
		}
		if size1 != size2 {
			t.Errorf("DecodeRune/DecodeRuneInString(%q) size mismatch %d/%d", s[i:], size1, size2)
			return
		}
		si += size1
	}
	j--
	for si = len(s); si > 0; {
		r1, size1 := DecodeLastRune(b[0:si])
		r2, size2 := DecodeLastRuneInString(s[0:si])
		if size1 != size2 {
			t.Errorf("DecodeLastRune/DecodeLastRuneInString(%q, %d) size mismatch %d/%d", s, si, size1, size2)
			return
		}
		if r1 != index[j].r {
			t.Errorf("DecodeLastRune(%q, %d) = %#04x, want %#04x", s, si, r1, index[j].r)
			return
		}
		if r2 != index[j].r {
			t.Errorf("DecodeLastRuneInString(%q, %d) = %#04x, want %#04x", s, si, r2, index[j].r)
			return
		}
		si -= size1
		if si != index[j].index {
			t.Errorf("DecodeLastRune(%q) index mismatch at %d, want %d", s, si, index[j].index)
			return
		}
		j--
	}
	if si != 0 {
		t.Errorf("DecodeLastRune(%q) finished at %d, not 0", s, si)
	}
}

// Check that negative runes encode as U+FFFD.
func TestNegativeRune(t *testing.T) {
	errorbuf := make([]byte, UTFMax)
	errorbuf = errorbuf[0:EncodeRune(errorbuf, RuneError)]
	buf := make([]byte, UTFMax)
	buf = buf[0:EncodeRune(buf, -1)]
	if !bytes.Equal(buf, errorbuf) {
		t.Errorf("incorrect encoding [% x] for -1; expected [% x]", buf, errorbuf)
	}
}

type RuneCountTest struct {
	in  string
	out int
}

var runecounttests = []RuneCountTest{
	{"abcd", 4},
	{"☺☻☹", 3},
	{"1,2,3,4", 7},
	{"\xe2\x00", 2},
}

func TestRuneCount(t *testing.T) {
	for _, tt := range runecounttests {
		if out := RuneCountInString(tt.in); out != tt.out {
			t.Errorf("RuneCountInString(%q) = %d, want %d", tt.in, out, tt.out)
		}
		if out := RuneCount([]byte(tt.in)); out != tt.out {
			t.Errorf("RuneCount(%q) = %d, want %d", tt.in, out, tt.out)
		}
	}
}

type RuneLenTest struct {
	r    rune
	size int
}

var runelentests = []RuneLenTest{
	{0, 1},
	{'e', 1},
	{'é', 2},
	{'☺', 3},
	{RuneError, 3},
	{MaxRune, 4},
	{0xD800, -1},
	{0xDFFF, -1},
	{MaxRune + 1, -1},
	{-1, -1},
}

func TestRuneLen(t *testing.T) {
	for _, tt := range runelentests {
		if size := RuneLen(tt.r); size != tt.size {
			t.Errorf("RuneLen(%#U) = %d, want %d", tt.r, size, tt.size)
		}
	}
}

type ValidTest struct {
	in  string
	out bool
}

var validTests = []ValidTest{
	{"", true},
	{"a", true},
	{"abc", true},
	{"Ж", true},
	{"ЖЖ", true},
	{"брэд-ЛГТМ", true},
	{"☺☻☹", true},
	{string([]byte{66, 250}), false},
	{string([]byte{66, 250, 67}), false},
	{"a\uFFFDb", true},
	{string("\xF4\x8F\xBF\xBF"), true},      // U+10FFFF
	{string("\xF4\x90\x80\x80"), false},     // U+10FFFF+1; out of range
	{string("\xF7\xBF\xBF\xBF"), false},     // 0x1FFFFF; out of range
	{string("\xFB\xBF\xBF\xBF\xBF"), false}, // 0x3FFFFFF; out of range
	{string("\xc0\x80"), false},             // U+0000 encoded in two bytes: incorrect
	{string("\xed\xa0\x80"), false},         // U+D800 high surrogate (sic)
	{string("\xed\xbf\xbf"), false},         // U+DFFF low surrogate (sic)
}

func TestValid(t *testing.T) {
	for _, tt := range validTests {
		if Valid([]byte(tt.in)) != tt.out {
			t.Errorf("Valid(%q) = %v; want %v", tt.in, !tt.out, tt.out)
		}
		if ValidString(tt.in) != tt.out {
			t.Errorf("ValidString(%q) = %v; want %v", tt.in, !tt.out, tt.out)
		}
	}
}

type ValidRuneTest struct {
	r  rune
	ok bool
}

var validrunetests = []ValidRuneTest{
	{0, true},
	{'e', true},
	{'é', true},
	{'☺', true},
	{RuneError, true},
	{MaxRune, true},
	{0xD7FF, true},
	{0xD800, false},
	{0xDFFF, false},
	{0xE000, true},
	{MaxRune + 1, false},
	{-1, false},
}

func TestValidRune(t *testing.T) {
	for _, tt := range validrunetests {
		if ok := ValidRune(tt.r); ok != tt.ok {
			t.Errorf("ValidRune(%#U) = %t, want %t", tt.r, ok, tt.ok)
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
		EncodeRune(buf, 'a')
	}
}

func BenchmarkEncodeJapaneseRune(b *testing.B) {
	buf := make([]byte, UTFMax)
	for i := 0; i < b.N; i++ {
		EncodeRune(buf, '本')
	}
}

func BenchmarkDecodeASCIIRune(b *testing.B) {
	a := []byte{'a'}
	for i := 0; i < b.N; i++ {
		DecodeRune(a)
	}
}

func BenchmarkDecodeJapaneseRune(b *testing.B) {
	nihon := []byte("本")
	for i := 0; i < b.N; i++ {
		DecodeRune(nihon)
	}
}
