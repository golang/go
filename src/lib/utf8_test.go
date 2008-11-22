// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf8

import (
	"fmt";
	"syscall";
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

func CEscape(s *[]byte) string {
	t := "\"";
	for i := 0; i < len(s); i++ {
		switch {
		case s[i] == '\\' || s[i] == '"':
			t += `\`;
			t += string(s[i]);
		case s[i] == '\n':
			t += `\n`;
		case s[i] == '\t':
			t += `\t`;
		case ' ' <= s[i] && s[i] <= '~':
			t += string(s[i]);
		default:
			t += fmt.sprintf(`\x%02x`, s[i]);
		}
	}
	t += "\"";
	return t;
}

func Bytes(s string) *[]byte {
	b := new([]byte, len(s)+1);
	if !syscall.StringToBytes(b, s) {
		panic("StringToBytes failed");
	}
	return b[0:len(s)];
}

export func TestFullRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i];
		b := Bytes(m.str);
		if !utf8.FullRune(b) {
			t.Errorf("FullRune(%s) (rune %04x) = false, want true", CEscape(b), m.rune);
		}
		if b1 := b[0:len(b)-1]; utf8.FullRune(b1) {
			t.Errorf("FullRune(%s) = true, want false", CEscape(b1));
		}
	}
}

func EqualBytes(a, b *[]byte) bool {
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

export func TestEncodeRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i];
		b := Bytes(m.str);
		var buf [10]byte;
		n := utf8.EncodeRune(m.rune, &buf);
		b1 := (&buf)[0:n];
		if !EqualBytes(b, b1) {
			t.Errorf("EncodeRune(0x%04x) = %s want %s", m.rune, CEscape(b1), CEscape(b));
		}
	}
}

export func TestDecodeRune(t *testing.T) {
	for i := 0; i < len(utf8map); i++ {
		m := utf8map[i];
		b := Bytes(m.str);
		rune, size := utf8.DecodeRune(b);
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRune(%s) = 0x%04x, %d want 0x%04x, %d", CEscape(b), rune, size, m.rune, len(b));
		}

		// there's an extra byte that Bytes left behind - make sure trailing byte works
		rune, size = utf8.DecodeRune(b[0:cap(b)]);
		if rune != m.rune || size != len(b) {
			t.Errorf("DecodeRune(%s) = 0x%04x, %d want 0x%04x, %d", CEscape(b), rune, size, m.rune, len(b));
		}

		// make sure missing bytes fail
		rune, size = utf8.DecodeRune(b[0:len(b)-1]);
		wantsize := 1;
		if wantsize >= len(b) {
			wantsize = 0;
		}
		if rune != RuneError || size != wantsize {
			t.Errorf("DecodeRune(%s) = 0x%04x, %d want 0x%04x, %d", CEscape(b[0:len(b)-1]), rune, size, RuneError, wantsize);
		}

		// make sure bad sequences fail
		if len(b) == 1 {
			b[0] = 0x80;
		} else {
			b[len(b)-1] = 0x7F;
		}
		rune, size = utf8.DecodeRune(b);
		if rune != RuneError || size != 1 {
			t.Errorf("DecodeRune(%s) = 0x%04x, %d want 0x%04x, %d", CEscape(b), rune, size, RuneError, 1);
		}
	}
}
