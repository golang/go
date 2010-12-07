// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf8_test

import (
	"bytes"
	"testing"
	. "utf8"
)

type Utf8Map struct {
	rune int
	str  string
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
	{0xfffe, "\xef\xbf\xbe"},
	{0xffff, "\xef\xbf\xbf"},
	{0x10000, "\xf0\x90\x80\x80"},
	{0x10001, "\xf0\x90\x80\x81"},
	{0x10fffe, "\xf4\x8f\xbf\xbe"},
	{0x10ffff, "\xf4\x8f\xbf\xbf"},
	{0xFFFD, "\xef\xbf\xbd"},
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
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i]
		b := []byte(m.str)
		if !FullRune(b) {
			t.Errorf("FullRune(%q) (%U) = false, want true", b, m.rune)
		}
		s := m.str
		if !FullRuneInString(s) {
			t.Errorf("FullRuneInString(%q) (%U) = false, want true", s, m.rune)
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
		b := []byte(m.str)
		var buf [10]byte
		n := EncodeRune(buf[0:], m.rune)
		b1 := buf[0:n]
		if !bytes.Equal(b, b1) {
			t.Errorf("EncodeRune(%#04x) = %q want %q", m.rune, b1, b)
		}
	}
}

func TestDecodeRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i]
		b := []byte(m.str)
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
		runes := []int(ts)
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
		rune  int
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
		rune1, size1 := DecodeRune(b[i:])
		if r != rune1 {
			t.Errorf("DecodeRune(%q) = %#04x, want %#04x", s[i:], rune1, r)
			return
		}
		rune2, size2 := DecodeRuneInString(s[i:])
		if r != rune2 {
			t.Errorf("DecodeRuneInString(%q) = %#04x, want %#04x", s[i:], rune2, r)
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
		rune1, size1 := DecodeLastRune(b[0:si])
		rune2, size2 := DecodeLastRuneInString(s[0:si])
		if size1 != size2 {
			t.Errorf("DecodeLastRune/DecodeLastRuneInString(%q, %d) size mismatch %d/%d", s, si, size1, size2)
			return
		}
		if rune1 != index[j].rune {
			t.Errorf("DecodeLastRune(%q, %d) = %#04x, want %#04x", s, si, rune1, index[j].rune)
			return
		}
		if rune2 != index[j].rune {
			t.Errorf("DecodeLastRuneInString(%q, %d) = %#04x, want %#04x", s, si, rune2, index[j].rune)
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
	for i := 0; i < len(runecounttests); i++ {
		tt := runecounttests[i]
		if out := RuneCountInString(tt.in); out != tt.out {
			t.Errorf("RuneCountInString(%q) = %d, want %d", tt.in, out, tt.out)
		}
		if out := RuneCount([]byte(tt.in)); out != tt.out {
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
