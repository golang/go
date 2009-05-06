// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf8

import (
	"fmt";
	"io";
	"testing";
	"utf8";
)

type Utf8Map struct {
	rune int;
	str string;
}

var utf8map = []Utf8Map {
	Utf8Map{ 0x0000, "\x00" },
	Utf8Map{ 0x0001, "\x01" },
	Utf8Map{ 0x007e, "\x7e" },
	Utf8Map{ 0x007f, "\x7f" },
	Utf8Map{ 0x0080, "\xc2\x80" },
	Utf8Map{ 0x0081, "\xc2\x81" },
	Utf8Map{ 0x00bf, "\xc2\xbf" },
	Utf8Map{ 0x00c0, "\xc3\x80" },
	Utf8Map{ 0x00c1, "\xc3\x81" },
	Utf8Map{ 0x00c8, "\xc3\x88" },
	Utf8Map{ 0x00d0, "\xc3\x90" },
	Utf8Map{ 0x00e0, "\xc3\xa0" },
	Utf8Map{ 0x00f0, "\xc3\xb0" },
	Utf8Map{ 0x00f8, "\xc3\xb8" },
	Utf8Map{ 0x00ff, "\xc3\xbf" },
	Utf8Map{ 0x0100, "\xc4\x80" },
	Utf8Map{ 0x07ff, "\xdf\xbf" },
	Utf8Map{ 0x0800, "\xe0\xa0\x80" },
	Utf8Map{ 0x0801, "\xe0\xa0\x81" },
	Utf8Map{ 0xfffe, "\xef\xbf\xbe" },
	Utf8Map{ 0xffff, "\xef\xbf\xbf" },
	Utf8Map{ 0x10000, "\xf0\x90\x80\x80" },
	Utf8Map{ 0x10001, "\xf0\x90\x80\x81" },
	Utf8Map{ 0x10fffe, "\xf4\x8f\xbf\xbe" },
	Utf8Map{ 0x10ffff, "\xf4\x8f\xbf\xbf" },
}

// io.StringBytes with one extra byte at end
func bytes(s string) []byte {
	s += "\x00";
	b := io.StringBytes(s);
	return b[0:len(s)-1];
}

func TestFullRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i];
		b := bytes(m.str);
		if !utf8.FullRune(b) {
			t.Errorf("FullRune(%q) (rune %04x) = false, want true", b, m.rune);
		}
		s := "xx"+m.str;
		if !utf8.FullRuneInString(s, 2) {
			t.Errorf("FullRuneInString(%q, 2) (rune %04x) = false, want true", s, m.rune);
		}
		b1 := b[0:len(b)-1];
		if utf8.FullRune(b1) {
			t.Errorf("FullRune(%q) = true, want false", b1);
		}
		s1 := "xxx"+string(b1);
		if utf8.FullRuneInString(s1, 3) {
			t.Errorf("FullRune(%q, 3) = true, want false", s1);
		}
	}
}

func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false;
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false;
		}
	}
	return true;
}

func TestEncodeRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i];
		b := bytes(m.str);
		var buf [10]byte;
		n := utf8.EncodeRune(m.rune, &buf);
		b1 := buf[0:n];
		if !equalBytes(b, b1) {
			t.Errorf("EncodeRune(0x%04x) = %q want %q", m.rune, b1, b);
		}
	}
}

func TestDecodeRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i];
		b := bytes(m.str);
		rune, size := utf8.DecodeRune(b);
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRune(%q) = 0x%04x, %d want 0x%04x, %d", b, rune, size, m.rune, len(b));
		}
		s := "xx"+m.str;
		rune, size = utf8.DecodeRuneInString(s, 2);
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRune(%q, 2) = 0x%04x, %d want 0x%04x, %d", s, rune, size, m.rune, len(b));
		}

		// there's an extra byte that bytes left behind - make sure trailing byte works
		rune, size = utf8.DecodeRune(b[0:cap(b)]);
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRune(%q) = 0x%04x, %d want 0x%04x, %d", b, rune, size, m.rune, len(b));
		}
		s = "x"+m.str+"\x00";
		rune, size = utf8.DecodeRuneInString(s, 1);
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRuneInString(%q, 1) = 0x%04x, %d want 0x%04x, %d", s, rune, size, m.rune, len(b));
		}

		// make sure missing bytes fail
		wantsize := 1;
		if wantsize >= len(b) {
			wantsize = 0;
		}
		rune, size = utf8.DecodeRune(b[0:len(b)-1]);
		if rune != RuneError || size != wantsize {
			t.Errorf("DecodeRune(%q) = 0x%04x, %d want 0x%04x, %d", b[0:len(b)-1], rune, size, RuneError, wantsize);
		}
		s = "xxx"+m.str[0:len(m.str)-1];
		rune, size = utf8.DecodeRuneInString(s, 3);
		if rune != RuneError || size != wantsize {
			t.Errorf("DecodeRuneInString(%q, 3) = 0x%04x, %d want 0x%04x, %d", s, rune, size, RuneError, wantsize);
		}

		// make sure bad sequences fail
		if len(b) == 1 {
			b[0] = 0x80;
		} else {
			b[len(b)-1] = 0x7F;
		}
		rune, size = utf8.DecodeRune(b);
		if rune != RuneError || size != 1 {
			t.Errorf("DecodeRune(%q) = 0x%04x, %d want 0x%04x, %d", b, rune, size, RuneError, 1);
		}
		s = "xxxx"+string(b);
		rune, size = utf8.DecodeRune(b);
		if rune != RuneError || size != 1 {
			t.Errorf("DecodeRuneInString(%q, 4) = 0x%04x, %d want 0x%04x, %d", s, rune, size, RuneError, 1);
		}
	}
}

type RuneCountTest struct {
	in string;
	out int;
}
var runecounttests = []RuneCountTest {
	RuneCountTest{ "abcd", 4 },
	RuneCountTest{ "☺☻☹", 3 },
	RuneCountTest{ "1,2,3,4", 7 },
	RuneCountTest{ "\xe2\x00", 2 },
}
func TestRuneCount(t *testing.T) {
	for i := 0; i < len(runecounttests); i++ {
		tt := runecounttests[i];
		if out := utf8.RuneCountInString(tt.in); out != tt.out {
			t.Errorf("RuneCountInString(%q) = %d, want %d", tt.in, out, tt.out);
		}
		if out := utf8.RuneCount(bytes(tt.in)); out != tt.out {
			t.Errorf("RuneCount(%q) = %d, want %d", tt.in, out, tt.out);
		}
	}
}
