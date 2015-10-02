// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"fmt"
	"testing"
)

var stringTests = []struct {
	in   string
	out  string
	base int
	val  int64
	ok   bool
}{
	{in: "", ok: false},
	{in: "a", ok: false},
	{in: "z", ok: false},
	{in: "+", ok: false},
	{in: "-", ok: false},
	{in: "0b", ok: false},
	{in: "0x", ok: false},
	{in: "2", base: 2, ok: false},
	{in: "0b2", base: 0, ok: false},
	{in: "08", ok: false},
	{in: "8", base: 8, ok: false},
	{in: "0xg", base: 0, ok: false},
	{in: "g", base: 16, ok: false},
	{"0", "0", 0, 0, true},
	{"0", "0", 10, 0, true},
	{"0", "0", 16, 0, true},
	{"+0", "0", 0, 0, true},
	{"-0", "0", 0, 0, true},
	{"10", "10", 0, 10, true},
	{"10", "10", 10, 10, true},
	{"10", "10", 16, 16, true},
	{"-10", "-10", 16, -16, true},
	{"+10", "10", 16, 16, true},
	{"0x10", "16", 0, 16, true},
	{in: "0x10", base: 16, ok: false},
	{"-0x10", "-16", 0, -16, true},
	{"+0x10", "16", 0, 16, true},
	{"00", "0", 0, 0, true},
	{"0", "0", 8, 0, true},
	{"07", "7", 0, 7, true},
	{"7", "7", 8, 7, true},
	{"023", "19", 0, 19, true},
	{"23", "23", 8, 19, true},
	{"cafebabe", "cafebabe", 16, 0xcafebabe, true},
	{"0b0", "0", 0, 0, true},
	{"-111", "-111", 2, -7, true},
	{"-0b111", "-7", 0, -7, true},
	{"0b1001010111", "599", 0, 0x257, true},
	{"1001010111", "1001010111", 2, 0x257, true},
}

func format(base int) string {
	switch base {
	case 2:
		return "%b"
	case 8:
		return "%o"
	case 16:
		return "%x"
	}
	return "%d"
}

func TestGetString(t *testing.T) {
	z := new(Int)
	for i, test := range stringTests {
		if !test.ok {
			continue
		}
		z.SetInt64(test.val)

		if test.base == 10 {
			s := z.String()
			if s != test.out {
				t.Errorf("#%da got %s; want %s", i, s, test.out)
			}
		}

		s := fmt.Sprintf(format(test.base), z)
		if s != test.out {
			t.Errorf("#%db got %s; want %s", i, s, test.out)
		}
	}
}

func TestSetString(t *testing.T) {
	tmp := new(Int)
	for i, test := range stringTests {
		// initialize to a non-zero value so that issues with parsing
		// 0 are detected
		tmp.SetInt64(1234567890)
		n1, ok1 := new(Int).SetString(test.in, test.base)
		n2, ok2 := tmp.SetString(test.in, test.base)
		expected := NewInt(test.val)
		if ok1 != test.ok || ok2 != test.ok {
			t.Errorf("#%d (input '%s') ok incorrect (should be %t)", i, test.in, test.ok)
			continue
		}
		if !ok1 {
			if n1 != nil {
				t.Errorf("#%d (input '%s') n1 != nil", i, test.in)
			}
			continue
		}
		if !ok2 {
			if n2 != nil {
				t.Errorf("#%d (input '%s') n2 != nil", i, test.in)
			}
			continue
		}

		if ok1 && !isNormalized(n1) {
			t.Errorf("#%d (input '%s'): %v is not normalized", i, test.in, *n1)
		}
		if ok2 && !isNormalized(n2) {
			t.Errorf("#%d (input '%s'): %v is not normalized", i, test.in, *n2)
		}

		if n1.Cmp(expected) != 0 {
			t.Errorf("#%d (input '%s') got: %s want: %d", i, test.in, n1, test.val)
		}
		if n2.Cmp(expected) != 0 {
			t.Errorf("#%d (input '%s') got: %s want: %d", i, test.in, n2, test.val)
		}
	}
}

var formatTests = []struct {
	input  string
	format string
	output string
}{
	{"<nil>", "%x", "<nil>"},
	{"<nil>", "%#x", "<nil>"},
	{"<nil>", "%#y", "%!y(big.Int=<nil>)"},

	{"10", "%b", "1010"},
	{"10", "%o", "12"},
	{"10", "%d", "10"},
	{"10", "%v", "10"},
	{"10", "%x", "a"},
	{"10", "%X", "A"},
	{"-10", "%X", "-A"},
	{"10", "%y", "%!y(big.Int=10)"},
	{"-10", "%y", "%!y(big.Int=-10)"},

	{"10", "%#b", "1010"},
	{"10", "%#o", "012"},
	{"10", "%#d", "10"},
	{"10", "%#v", "10"},
	{"10", "%#x", "0xa"},
	{"10", "%#X", "0XA"},
	{"-10", "%#X", "-0XA"},
	{"10", "%#y", "%!y(big.Int=10)"},
	{"-10", "%#y", "%!y(big.Int=-10)"},

	{"1234", "%d", "1234"},
	{"1234", "%3d", "1234"},
	{"1234", "%4d", "1234"},
	{"-1234", "%d", "-1234"},
	{"1234", "% 5d", " 1234"},
	{"1234", "%+5d", "+1234"},
	{"1234", "%-5d", "1234 "},
	{"1234", "%x", "4d2"},
	{"1234", "%X", "4D2"},
	{"-1234", "%3x", "-4d2"},
	{"-1234", "%4x", "-4d2"},
	{"-1234", "%5x", " -4d2"},
	{"-1234", "%-5x", "-4d2 "},
	{"1234", "%03d", "1234"},
	{"1234", "%04d", "1234"},
	{"1234", "%05d", "01234"},
	{"1234", "%06d", "001234"},
	{"-1234", "%06d", "-01234"},
	{"1234", "%+06d", "+01234"},
	{"1234", "% 06d", " 01234"},
	{"1234", "%-6d", "1234  "},
	{"1234", "%-06d", "1234  "},
	{"-1234", "%-06d", "-1234 "},

	{"1234", "%.3d", "1234"},
	{"1234", "%.4d", "1234"},
	{"1234", "%.5d", "01234"},
	{"1234", "%.6d", "001234"},
	{"-1234", "%.3d", "-1234"},
	{"-1234", "%.4d", "-1234"},
	{"-1234", "%.5d", "-01234"},
	{"-1234", "%.6d", "-001234"},

	{"1234", "%8.3d", "    1234"},
	{"1234", "%8.4d", "    1234"},
	{"1234", "%8.5d", "   01234"},
	{"1234", "%8.6d", "  001234"},
	{"-1234", "%8.3d", "   -1234"},
	{"-1234", "%8.4d", "   -1234"},
	{"-1234", "%8.5d", "  -01234"},
	{"-1234", "%8.6d", " -001234"},

	{"1234", "%+8.3d", "   +1234"},
	{"1234", "%+8.4d", "   +1234"},
	{"1234", "%+8.5d", "  +01234"},
	{"1234", "%+8.6d", " +001234"},
	{"-1234", "%+8.3d", "   -1234"},
	{"-1234", "%+8.4d", "   -1234"},
	{"-1234", "%+8.5d", "  -01234"},
	{"-1234", "%+8.6d", " -001234"},

	{"1234", "% 8.3d", "    1234"},
	{"1234", "% 8.4d", "    1234"},
	{"1234", "% 8.5d", "   01234"},
	{"1234", "% 8.6d", "  001234"},
	{"-1234", "% 8.3d", "   -1234"},
	{"-1234", "% 8.4d", "   -1234"},
	{"-1234", "% 8.5d", "  -01234"},
	{"-1234", "% 8.6d", " -001234"},

	{"1234", "%.3x", "4d2"},
	{"1234", "%.4x", "04d2"},
	{"1234", "%.5x", "004d2"},
	{"1234", "%.6x", "0004d2"},
	{"-1234", "%.3x", "-4d2"},
	{"-1234", "%.4x", "-04d2"},
	{"-1234", "%.5x", "-004d2"},
	{"-1234", "%.6x", "-0004d2"},

	{"1234", "%8.3x", "     4d2"},
	{"1234", "%8.4x", "    04d2"},
	{"1234", "%8.5x", "   004d2"},
	{"1234", "%8.6x", "  0004d2"},
	{"-1234", "%8.3x", "    -4d2"},
	{"-1234", "%8.4x", "   -04d2"},
	{"-1234", "%8.5x", "  -004d2"},
	{"-1234", "%8.6x", " -0004d2"},

	{"1234", "%+8.3x", "    +4d2"},
	{"1234", "%+8.4x", "   +04d2"},
	{"1234", "%+8.5x", "  +004d2"},
	{"1234", "%+8.6x", " +0004d2"},
	{"-1234", "%+8.3x", "    -4d2"},
	{"-1234", "%+8.4x", "   -04d2"},
	{"-1234", "%+8.5x", "  -004d2"},
	{"-1234", "%+8.6x", " -0004d2"},

	{"1234", "% 8.3x", "     4d2"},
	{"1234", "% 8.4x", "    04d2"},
	{"1234", "% 8.5x", "   004d2"},
	{"1234", "% 8.6x", "  0004d2"},
	{"1234", "% 8.7x", " 00004d2"},
	{"1234", "% 8.8x", " 000004d2"},
	{"-1234", "% 8.3x", "    -4d2"},
	{"-1234", "% 8.4x", "   -04d2"},
	{"-1234", "% 8.5x", "  -004d2"},
	{"-1234", "% 8.6x", " -0004d2"},
	{"-1234", "% 8.7x", "-00004d2"},
	{"-1234", "% 8.8x", "-000004d2"},

	{"1234", "%-8.3d", "1234    "},
	{"1234", "%-8.4d", "1234    "},
	{"1234", "%-8.5d", "01234   "},
	{"1234", "%-8.6d", "001234  "},
	{"1234", "%-8.7d", "0001234 "},
	{"1234", "%-8.8d", "00001234"},
	{"-1234", "%-8.3d", "-1234   "},
	{"-1234", "%-8.4d", "-1234   "},
	{"-1234", "%-8.5d", "-01234  "},
	{"-1234", "%-8.6d", "-001234 "},
	{"-1234", "%-8.7d", "-0001234"},
	{"-1234", "%-8.8d", "-00001234"},

	{"16777215", "%b", "111111111111111111111111"}, // 2**24 - 1

	{"0", "%.d", ""},
	{"0", "%.0d", ""},
	{"0", "%3.d", ""},
}

func TestFormat(t *testing.T) {
	for i, test := range formatTests {
		var x *Int
		if test.input != "<nil>" {
			var ok bool
			x, ok = new(Int).SetString(test.input, 0)
			if !ok {
				t.Errorf("#%d failed reading input %s", i, test.input)
			}
		}
		output := fmt.Sprintf(test.format, x)
		if output != test.output {
			t.Errorf("#%d got %q; want %q, {%q, %q, %q}", i, output, test.output, test.input, test.format, test.output)
		}
	}
}

var scanTests = []struct {
	input     string
	format    string
	output    string
	remaining int
}{
	{"1010", "%b", "10", 0},
	{"0b1010", "%v", "10", 0},
	{"12", "%o", "10", 0},
	{"012", "%v", "10", 0},
	{"10", "%d", "10", 0},
	{"10", "%v", "10", 0},
	{"a", "%x", "10", 0},
	{"0xa", "%v", "10", 0},
	{"A", "%X", "10", 0},
	{"-A", "%X", "-10", 0},
	{"+0b1011001", "%v", "89", 0},
	{"0xA", "%v", "10", 0},
	{"0 ", "%v", "0", 1},
	{"2+3", "%v", "2", 2},
	{"0XABC 12", "%v", "2748", 3},
}

func TestScan(t *testing.T) {
	var buf bytes.Buffer
	for i, test := range scanTests {
		x := new(Int)
		buf.Reset()
		buf.WriteString(test.input)
		if _, err := fmt.Fscanf(&buf, test.format, x); err != nil {
			t.Errorf("#%d error: %s", i, err)
		}
		if x.String() != test.output {
			t.Errorf("#%d got %s; want %s", i, x.String(), test.output)
		}
		if buf.Len() != test.remaining {
			t.Errorf("#%d got %d bytes remaining; want %d", i, buf.Len(), test.remaining)
		}
	}
}
