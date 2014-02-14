// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"math/rand"
	"testing"
	"testing/quick"
)

func isNormalized(x *Int) bool {
	if len(x.abs) == 0 {
		return !x.neg
	}
	// len(x.abs) > 0
	return x.abs[len(x.abs)-1] != 0
}

type funZZ func(z, x, y *Int) *Int
type argZZ struct {
	z, x, y *Int
}

var sumZZ = []argZZ{
	{NewInt(0), NewInt(0), NewInt(0)},
	{NewInt(1), NewInt(1), NewInt(0)},
	{NewInt(1111111110), NewInt(123456789), NewInt(987654321)},
	{NewInt(-1), NewInt(-1), NewInt(0)},
	{NewInt(864197532), NewInt(-123456789), NewInt(987654321)},
	{NewInt(-1111111110), NewInt(-123456789), NewInt(-987654321)},
}

var prodZZ = []argZZ{
	{NewInt(0), NewInt(0), NewInt(0)},
	{NewInt(0), NewInt(1), NewInt(0)},
	{NewInt(1), NewInt(1), NewInt(1)},
	{NewInt(-991 * 991), NewInt(991), NewInt(-991)},
	// TODO(gri) add larger products
}

func TestSignZ(t *testing.T) {
	var zero Int
	for _, a := range sumZZ {
		s := a.z.Sign()
		e := a.z.Cmp(&zero)
		if s != e {
			t.Errorf("got %d; want %d for z = %v", s, e, a.z)
		}
	}
}

func TestSetZ(t *testing.T) {
	for _, a := range sumZZ {
		var z Int
		z.Set(a.z)
		if !isNormalized(&z) {
			t.Errorf("%v is not normalized", z)
		}
		if (&z).Cmp(a.z) != 0 {
			t.Errorf("got z = %v; want %v", z, a.z)
		}
	}
}

func TestAbsZ(t *testing.T) {
	var zero Int
	for _, a := range sumZZ {
		var z Int
		z.Abs(a.z)
		var e Int
		e.Set(a.z)
		if e.Cmp(&zero) < 0 {
			e.Sub(&zero, &e)
		}
		if z.Cmp(&e) != 0 {
			t.Errorf("got z = %v; want %v", z, e)
		}
	}
}

func testFunZZ(t *testing.T, msg string, f funZZ, a argZZ) {
	var z Int
	f(&z, a.x, a.y)
	if !isNormalized(&z) {
		t.Errorf("%s%v is not normalized", msg, z)
	}
	if (&z).Cmp(a.z) != 0 {
		t.Errorf("%s%+v\n\tgot z = %v; want %v", msg, a, &z, a.z)
	}
}

func TestSumZZ(t *testing.T) {
	AddZZ := func(z, x, y *Int) *Int { return z.Add(x, y) }
	SubZZ := func(z, x, y *Int) *Int { return z.Sub(x, y) }
	for _, a := range sumZZ {
		arg := a
		testFunZZ(t, "AddZZ", AddZZ, arg)

		arg = argZZ{a.z, a.y, a.x}
		testFunZZ(t, "AddZZ symmetric", AddZZ, arg)

		arg = argZZ{a.x, a.z, a.y}
		testFunZZ(t, "SubZZ", SubZZ, arg)

		arg = argZZ{a.y, a.z, a.x}
		testFunZZ(t, "SubZZ symmetric", SubZZ, arg)
	}
}

func TestProdZZ(t *testing.T) {
	MulZZ := func(z, x, y *Int) *Int { return z.Mul(x, y) }
	for _, a := range prodZZ {
		arg := a
		testFunZZ(t, "MulZZ", MulZZ, arg)

		arg = argZZ{a.z, a.y, a.x}
		testFunZZ(t, "MulZZ symmetric", MulZZ, arg)
	}
}

// mulBytes returns x*y via grade school multiplication. Both inputs
// and the result are assumed to be in big-endian representation (to
// match the semantics of Int.Bytes and Int.SetBytes).
func mulBytes(x, y []byte) []byte {
	z := make([]byte, len(x)+len(y))

	// multiply
	k0 := len(z) - 1
	for j := len(y) - 1; j >= 0; j-- {
		d := int(y[j])
		if d != 0 {
			k := k0
			carry := 0
			for i := len(x) - 1; i >= 0; i-- {
				t := int(z[k]) + int(x[i])*d + carry
				z[k], carry = byte(t), t>>8
				k--
			}
			z[k] = byte(carry)
		}
		k0--
	}

	// normalize (remove leading 0's)
	i := 0
	for i < len(z) && z[i] == 0 {
		i++
	}

	return z[i:]
}

func checkMul(a, b []byte) bool {
	var x, y, z1 Int
	x.SetBytes(a)
	y.SetBytes(b)
	z1.Mul(&x, &y)

	var z2 Int
	z2.SetBytes(mulBytes(a, b))

	return z1.Cmp(&z2) == 0
}

func TestMul(t *testing.T) {
	if err := quick.Check(checkMul, nil); err != nil {
		t.Error(err)
	}
}

var mulRangesZ = []struct {
	a, b int64
	prod string
}{
	// entirely positive ranges are covered by mulRangesN
	{-1, 1, "0"},
	{-2, -1, "2"},
	{-3, -2, "6"},
	{-3, -1, "-6"},
	{1, 3, "6"},
	{-10, -10, "-10"},
	{0, -1, "1"},                      // empty range
	{-1, -100, "1"},                   // empty range
	{-1, 1, "0"},                      // range includes 0
	{-1e9, 0, "0"},                    // range includes 0
	{-1e9, 1e9, "0"},                  // range includes 0
	{-10, -1, "3628800"},              // 10!
	{-20, -2, "-2432902008176640000"}, // -20!
	{-99, -1,
		"-933262154439441526816992388562667004907159682643816214685929" +
			"638952175999932299156089414639761565182862536979208272237582" +
			"511852109168640000000000000000000000", // -99!
	},
}

func TestMulRangeZ(t *testing.T) {
	var tmp Int
	// test entirely positive ranges
	for i, r := range mulRangesN {
		prod := tmp.MulRange(int64(r.a), int64(r.b)).String()
		if prod != r.prod {
			t.Errorf("#%da: got %s; want %s", i, prod, r.prod)
		}
	}
	// test other ranges
	for i, r := range mulRangesZ {
		prod := tmp.MulRange(r.a, r.b).String()
		if prod != r.prod {
			t.Errorf("#%db: got %s; want %s", i, prod, r.prod)
		}
	}
}

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

// Examples from the Go Language Spec, section "Arithmetic operators"
var divisionSignsTests = []struct {
	x, y int64
	q, r int64 // T-division
	d, m int64 // Euclidian division
}{
	{5, 3, 1, 2, 1, 2},
	{-5, 3, -1, -2, -2, 1},
	{5, -3, -1, 2, -1, 2},
	{-5, -3, 1, -2, 2, 1},
	{1, 2, 0, 1, 0, 1},
	{8, 4, 2, 0, 2, 0},
}

func TestDivisionSigns(t *testing.T) {
	for i, test := range divisionSignsTests {
		x := NewInt(test.x)
		y := NewInt(test.y)
		q := NewInt(test.q)
		r := NewInt(test.r)
		d := NewInt(test.d)
		m := NewInt(test.m)

		q1 := new(Int).Quo(x, y)
		r1 := new(Int).Rem(x, y)
		if !isNormalized(q1) {
			t.Errorf("#%d Quo: %v is not normalized", i, *q1)
		}
		if !isNormalized(r1) {
			t.Errorf("#%d Rem: %v is not normalized", i, *r1)
		}
		if q1.Cmp(q) != 0 || r1.Cmp(r) != 0 {
			t.Errorf("#%d QuoRem: got (%s, %s), want (%s, %s)", i, q1, r1, q, r)
		}

		q2, r2 := new(Int).QuoRem(x, y, new(Int))
		if !isNormalized(q2) {
			t.Errorf("#%d Quo: %v is not normalized", i, *q2)
		}
		if !isNormalized(r2) {
			t.Errorf("#%d Rem: %v is not normalized", i, *r2)
		}
		if q2.Cmp(q) != 0 || r2.Cmp(r) != 0 {
			t.Errorf("#%d QuoRem: got (%s, %s), want (%s, %s)", i, q2, r2, q, r)
		}

		d1 := new(Int).Div(x, y)
		m1 := new(Int).Mod(x, y)
		if !isNormalized(d1) {
			t.Errorf("#%d Div: %v is not normalized", i, *d1)
		}
		if !isNormalized(m1) {
			t.Errorf("#%d Mod: %v is not normalized", i, *m1)
		}
		if d1.Cmp(d) != 0 || m1.Cmp(m) != 0 {
			t.Errorf("#%d DivMod: got (%s, %s), want (%s, %s)", i, d1, m1, d, m)
		}

		d2, m2 := new(Int).DivMod(x, y, new(Int))
		if !isNormalized(d2) {
			t.Errorf("#%d Div: %v is not normalized", i, *d2)
		}
		if !isNormalized(m2) {
			t.Errorf("#%d Mod: %v is not normalized", i, *m2)
		}
		if d2.Cmp(d) != 0 || m2.Cmp(m) != 0 {
			t.Errorf("#%d DivMod: got (%s, %s), want (%s, %s)", i, d2, m2, d, m)
		}
	}
}

func checkSetBytes(b []byte) bool {
	hex1 := hex.EncodeToString(new(Int).SetBytes(b).Bytes())
	hex2 := hex.EncodeToString(b)

	for len(hex1) < len(hex2) {
		hex1 = "0" + hex1
	}

	for len(hex1) > len(hex2) {
		hex2 = "0" + hex2
	}

	return hex1 == hex2
}

func TestSetBytes(t *testing.T) {
	if err := quick.Check(checkSetBytes, nil); err != nil {
		t.Error(err)
	}
}

func checkBytes(b []byte) bool {
	b2 := new(Int).SetBytes(b).Bytes()
	return bytes.Equal(b, b2)
}

func TestBytes(t *testing.T) {
	if err := quick.Check(checkSetBytes, nil); err != nil {
		t.Error(err)
	}
}

func checkQuo(x, y []byte) bool {
	u := new(Int).SetBytes(x)
	v := new(Int).SetBytes(y)

	if len(v.abs) == 0 {
		return true
	}

	r := new(Int)
	q, r := new(Int).QuoRem(u, v, r)

	if r.Cmp(v) >= 0 {
		return false
	}

	uprime := new(Int).Set(q)
	uprime.Mul(uprime, v)
	uprime.Add(uprime, r)

	return uprime.Cmp(u) == 0
}

var quoTests = []struct {
	x, y string
	q, r string
}{
	{
		"476217953993950760840509444250624797097991362735329973741718102894495832294430498335824897858659711275234906400899559094370964723884706254265559534144986498357",
		"9353930466774385905609975137998169297361893554149986716853295022578535724979483772383667534691121982974895531435241089241440253066816724367338287092081996",
		"50911",
		"1",
	},
	{
		"11510768301994997771168",
		"1328165573307167369775",
		"8",
		"885443715537658812968",
	},
}

func TestQuo(t *testing.T) {
	if err := quick.Check(checkQuo, nil); err != nil {
		t.Error(err)
	}

	for i, test := range quoTests {
		x, _ := new(Int).SetString(test.x, 10)
		y, _ := new(Int).SetString(test.y, 10)
		expectedQ, _ := new(Int).SetString(test.q, 10)
		expectedR, _ := new(Int).SetString(test.r, 10)

		r := new(Int)
		q, r := new(Int).QuoRem(x, y, r)

		if q.Cmp(expectedQ) != 0 || r.Cmp(expectedR) != 0 {
			t.Errorf("#%d got (%s, %s) want (%s, %s)", i, q, r, expectedQ, expectedR)
		}
	}
}

func TestQuoStepD6(t *testing.T) {
	// See Knuth, Volume 2, section 4.3.1, exercise 21. This code exercises
	// a code path which only triggers 1 in 10^{-19} cases.

	u := &Int{false, nat{0, 0, 1 + 1<<(_W-1), _M ^ (1 << (_W - 1))}}
	v := &Int{false, nat{5, 2 + 1<<(_W-1), 1 << (_W - 1)}}

	r := new(Int)
	q, r := new(Int).QuoRem(u, v, r)
	const expectedQ64 = "18446744073709551613"
	const expectedR64 = "3138550867693340382088035895064302439801311770021610913807"
	const expectedQ32 = "4294967293"
	const expectedR32 = "39614081266355540837921718287"
	if q.String() != expectedQ64 && q.String() != expectedQ32 ||
		r.String() != expectedR64 && r.String() != expectedR32 {
		t.Errorf("got (%s, %s) want (%s, %s) or (%s, %s)", q, r, expectedQ64, expectedR64, expectedQ32, expectedR32)
	}
}

var bitLenTests = []struct {
	in  string
	out int
}{
	{"-1", 1},
	{"0", 0},
	{"1", 1},
	{"2", 2},
	{"4", 3},
	{"0xabc", 12},
	{"0x8000", 16},
	{"0x80000000", 32},
	{"0x800000000000", 48},
	{"0x8000000000000000", 64},
	{"0x80000000000000000000", 80},
	{"-0x4000000000000000000000", 87},
}

func TestBitLen(t *testing.T) {
	for i, test := range bitLenTests {
		x, ok := new(Int).SetString(test.in, 0)
		if !ok {
			t.Errorf("#%d test input invalid: %s", i, test.in)
			continue
		}

		if n := x.BitLen(); n != test.out {
			t.Errorf("#%d got %d want %d", i, n, test.out)
		}
	}
}

var expTests = []struct {
	x, y, m string
	out     string
}{
	{"5", "-7", "", "1"},
	{"-5", "-7", "", "1"},
	{"5", "0", "", "1"},
	{"-5", "0", "", "1"},
	{"5", "1", "", "5"},
	{"-5", "1", "", "-5"},
	{"-2", "3", "2", "0"},
	{"5", "2", "", "25"},
	{"1", "65537", "2", "1"},
	{"0x8000000000000000", "2", "", "0x40000000000000000000000000000000"},
	{"0x8000000000000000", "2", "6719", "4944"},
	{"0x8000000000000000", "3", "6719", "5447"},
	{"0x8000000000000000", "1000", "6719", "1603"},
	{"0x8000000000000000", "1000000", "6719", "3199"},
	{"0x8000000000000000", "-1000000", "6719", "1"},
	{
		"2938462938472983472983659726349017249287491026512746239764525612965293865296239471239874193284792387498274256129746192347",
		"298472983472983471903246121093472394872319615612417471234712061",
		"29834729834729834729347290846729561262544958723956495615629569234729836259263598127342374289365912465901365498236492183464",
		"23537740700184054162508175125554701713153216681790245129157191391322321508055833908509185839069455749219131480588829346291",
	},
}

func TestExp(t *testing.T) {
	for i, test := range expTests {
		x, ok1 := new(Int).SetString(test.x, 0)
		y, ok2 := new(Int).SetString(test.y, 0)
		out, ok3 := new(Int).SetString(test.out, 0)

		var ok4 bool
		var m *Int

		if len(test.m) == 0 {
			m, ok4 = nil, true
		} else {
			m, ok4 = new(Int).SetString(test.m, 0)
		}

		if !ok1 || !ok2 || !ok3 || !ok4 {
			t.Errorf("#%d: error in input", i)
			continue
		}

		z1 := new(Int).Exp(x, y, m)
		if !isNormalized(z1) {
			t.Errorf("#%d: %v is not normalized", i, *z1)
		}
		if z1.Cmp(out) != 0 {
			t.Errorf("#%d: got %s want %s", i, z1, out)
		}

		if m == nil {
			// the result should be the same as for m == 0;
			// specifically, there should be no div-zero panic
			m = &Int{abs: nat{}} // m != nil && len(m.abs) == 0
			z2 := new(Int).Exp(x, y, m)
			if z2.Cmp(z1) != 0 {
				t.Errorf("#%d: got %s want %s", i, z1, z2)
			}
		}
	}
}

func checkGcd(aBytes, bBytes []byte) bool {
	x := new(Int)
	y := new(Int)
	a := new(Int).SetBytes(aBytes)
	b := new(Int).SetBytes(bBytes)

	d := new(Int).GCD(x, y, a, b)
	x.Mul(x, a)
	y.Mul(y, b)
	x.Add(x, y)

	return x.Cmp(d) == 0
}

var gcdTests = []struct {
	d, x, y, a, b string
}{
	// a <= 0 || b <= 0
	{"0", "0", "0", "0", "0"},
	{"0", "0", "0", "0", "7"},
	{"0", "0", "0", "11", "0"},
	{"0", "0", "0", "-77", "35"},
	{"0", "0", "0", "64515", "-24310"},
	{"0", "0", "0", "-64515", "-24310"},

	{"1", "-9", "47", "120", "23"},
	{"7", "1", "-2", "77", "35"},
	{"935", "-3", "8", "64515", "24310"},
	{"935000000000000000", "-3", "8", "64515000000000000000", "24310000000000000000"},
	{"1", "-221", "22059940471369027483332068679400581064239780177629666810348940098015901108344", "98920366548084643601728869055592650835572950932266967461790948584315647051443", "991"},

	// test early exit (after one Euclidean iteration) in binaryGCD
	{"1", "", "", "1", "98920366548084643601728869055592650835572950932266967461790948584315647051443"},
}

func testGcd(t *testing.T, d, x, y, a, b *Int) {
	var X *Int
	if x != nil {
		X = new(Int)
	}
	var Y *Int
	if y != nil {
		Y = new(Int)
	}

	D := new(Int).GCD(X, Y, a, b)
	if D.Cmp(d) != 0 {
		t.Errorf("GCD(%s, %s): got d = %s, want %s", a, b, D, d)
	}
	if x != nil && X.Cmp(x) != 0 {
		t.Errorf("GCD(%s, %s): got x = %s, want %s", a, b, X, x)
	}
	if y != nil && Y.Cmp(y) != 0 {
		t.Errorf("GCD(%s, %s): got y = %s, want %s", a, b, Y, y)
	}

	// binaryGCD requires a > 0 && b > 0
	if a.Sign() <= 0 || b.Sign() <= 0 {
		return
	}

	D.binaryGCD(a, b)
	if D.Cmp(d) != 0 {
		t.Errorf("binaryGcd(%s, %s): got d = %s, want %s", a, b, D, d)
	}
}

func TestGcd(t *testing.T) {
	for _, test := range gcdTests {
		d, _ := new(Int).SetString(test.d, 0)
		x, _ := new(Int).SetString(test.x, 0)
		y, _ := new(Int).SetString(test.y, 0)
		a, _ := new(Int).SetString(test.a, 0)
		b, _ := new(Int).SetString(test.b, 0)

		testGcd(t, d, nil, nil, a, b)
		testGcd(t, d, x, nil, a, b)
		testGcd(t, d, nil, y, a, b)
		testGcd(t, d, x, y, a, b)
	}

	quick.Check(checkGcd, nil)
}

var primes = []string{
	"2",
	"3",
	"5",
	"7",
	"11",

	"13756265695458089029",
	"13496181268022124907",
	"10953742525620032441",
	"17908251027575790097",

	// http://code.google.com/p/go/issues/detail?id=638
	"18699199384836356663",

	"98920366548084643601728869055592650835572950932266967461790948584315647051443",
	"94560208308847015747498523884063394671606671904944666360068158221458669711639",

	// http://primes.utm.edu/lists/small/small3.html
	"449417999055441493994709297093108513015373787049558499205492347871729927573118262811508386655998299074566974373711472560655026288668094291699357843464363003144674940345912431129144354948751003607115263071543163",
	"230975859993204150666423538988557839555560243929065415434980904258310530753006723857139742334640122533598517597674807096648905501653461687601339782814316124971547968912893214002992086353183070342498989426570593",
	"5521712099665906221540423207019333379125265462121169655563495403888449493493629943498064604536961775110765377745550377067893607246020694972959780839151452457728855382113555867743022746090187341871655890805971735385789993",
	"203956878356401977405765866929034577280193993314348263094772646453283062722701277632936616063144088173312372882677123879538709400158306567338328279154499698366071906766440037074217117805690872792848149112022286332144876183376326512083574821647933992961249917319836219304274280243803104015000563790123",
}

var composites = []string{
	"21284175091214687912771199898307297748211672914763848041968395774954376176754",
	"6084766654921918907427900243509372380954290099172559290432744450051395395951",
	"84594350493221918389213352992032324280367711247940675652888030554255915464401",
	"82793403787388584738507275144194252681",
}

func TestProbablyPrime(t *testing.T) {
	nreps := 20
	if testing.Short() {
		nreps = 1
	}
	for i, s := range primes {
		p, _ := new(Int).SetString(s, 10)
		if !p.ProbablyPrime(nreps) {
			t.Errorf("#%d prime found to be non-prime (%s)", i, s)
		}
	}

	for i, s := range composites {
		c, _ := new(Int).SetString(s, 10)
		if c.ProbablyPrime(nreps) {
			t.Errorf("#%d composite found to be prime (%s)", i, s)
		}
		if testing.Short() {
			break
		}
	}
}

type intShiftTest struct {
	in    string
	shift uint
	out   string
}

var rshTests = []intShiftTest{
	{"0", 0, "0"},
	{"-0", 0, "0"},
	{"0", 1, "0"},
	{"0", 2, "0"},
	{"1", 0, "1"},
	{"1", 1, "0"},
	{"1", 2, "0"},
	{"2", 0, "2"},
	{"2", 1, "1"},
	{"-1", 0, "-1"},
	{"-1", 1, "-1"},
	{"-1", 10, "-1"},
	{"-100", 2, "-25"},
	{"-100", 3, "-13"},
	{"-100", 100, "-1"},
	{"4294967296", 0, "4294967296"},
	{"4294967296", 1, "2147483648"},
	{"4294967296", 2, "1073741824"},
	{"18446744073709551616", 0, "18446744073709551616"},
	{"18446744073709551616", 1, "9223372036854775808"},
	{"18446744073709551616", 2, "4611686018427387904"},
	{"18446744073709551616", 64, "1"},
	{"340282366920938463463374607431768211456", 64, "18446744073709551616"},
	{"340282366920938463463374607431768211456", 128, "1"},
}

func TestRsh(t *testing.T) {
	for i, test := range rshTests {
		in, _ := new(Int).SetString(test.in, 10)
		expected, _ := new(Int).SetString(test.out, 10)
		out := new(Int).Rsh(in, test.shift)

		if !isNormalized(out) {
			t.Errorf("#%d: %v is not normalized", i, *out)
		}
		if out.Cmp(expected) != 0 {
			t.Errorf("#%d: got %s want %s", i, out, expected)
		}
	}
}

func TestRshSelf(t *testing.T) {
	for i, test := range rshTests {
		z, _ := new(Int).SetString(test.in, 10)
		expected, _ := new(Int).SetString(test.out, 10)
		z.Rsh(z, test.shift)

		if !isNormalized(z) {
			t.Errorf("#%d: %v is not normalized", i, *z)
		}
		if z.Cmp(expected) != 0 {
			t.Errorf("#%d: got %s want %s", i, z, expected)
		}
	}
}

var lshTests = []intShiftTest{
	{"0", 0, "0"},
	{"0", 1, "0"},
	{"0", 2, "0"},
	{"1", 0, "1"},
	{"1", 1, "2"},
	{"1", 2, "4"},
	{"2", 0, "2"},
	{"2", 1, "4"},
	{"2", 2, "8"},
	{"-87", 1, "-174"},
	{"4294967296", 0, "4294967296"},
	{"4294967296", 1, "8589934592"},
	{"4294967296", 2, "17179869184"},
	{"18446744073709551616", 0, "18446744073709551616"},
	{"9223372036854775808", 1, "18446744073709551616"},
	{"4611686018427387904", 2, "18446744073709551616"},
	{"1", 64, "18446744073709551616"},
	{"18446744073709551616", 64, "340282366920938463463374607431768211456"},
	{"1", 128, "340282366920938463463374607431768211456"},
}

func TestLsh(t *testing.T) {
	for i, test := range lshTests {
		in, _ := new(Int).SetString(test.in, 10)
		expected, _ := new(Int).SetString(test.out, 10)
		out := new(Int).Lsh(in, test.shift)

		if !isNormalized(out) {
			t.Errorf("#%d: %v is not normalized", i, *out)
		}
		if out.Cmp(expected) != 0 {
			t.Errorf("#%d: got %s want %s", i, out, expected)
		}
	}
}

func TestLshSelf(t *testing.T) {
	for i, test := range lshTests {
		z, _ := new(Int).SetString(test.in, 10)
		expected, _ := new(Int).SetString(test.out, 10)
		z.Lsh(z, test.shift)

		if !isNormalized(z) {
			t.Errorf("#%d: %v is not normalized", i, *z)
		}
		if z.Cmp(expected) != 0 {
			t.Errorf("#%d: got %s want %s", i, z, expected)
		}
	}
}

func TestLshRsh(t *testing.T) {
	for i, test := range rshTests {
		in, _ := new(Int).SetString(test.in, 10)
		out := new(Int).Lsh(in, test.shift)
		out = out.Rsh(out, test.shift)

		if !isNormalized(out) {
			t.Errorf("#%d: %v is not normalized", i, *out)
		}
		if in.Cmp(out) != 0 {
			t.Errorf("#%d: got %s want %s", i, out, in)
		}
	}
	for i, test := range lshTests {
		in, _ := new(Int).SetString(test.in, 10)
		out := new(Int).Lsh(in, test.shift)
		out.Rsh(out, test.shift)

		if !isNormalized(out) {
			t.Errorf("#%d: %v is not normalized", i, *out)
		}
		if in.Cmp(out) != 0 {
			t.Errorf("#%d: got %s want %s", i, out, in)
		}
	}
}

var int64Tests = []int64{
	0,
	1,
	-1,
	4294967295,
	-4294967295,
	4294967296,
	-4294967296,
	9223372036854775807,
	-9223372036854775807,
	-9223372036854775808,
}

func TestInt64(t *testing.T) {
	for i, testVal := range int64Tests {
		in := NewInt(testVal)
		out := in.Int64()

		if out != testVal {
			t.Errorf("#%d got %d want %d", i, out, testVal)
		}
	}
}

var uint64Tests = []uint64{
	0,
	1,
	4294967295,
	4294967296,
	8589934591,
	8589934592,
	9223372036854775807,
	9223372036854775808,
	18446744073709551615, // 1<<64 - 1
}

func TestUint64(t *testing.T) {
	in := new(Int)
	for i, testVal := range uint64Tests {
		in.SetUint64(testVal)
		out := in.Uint64()

		if out != testVal {
			t.Errorf("#%d got %d want %d", i, out, testVal)
		}

		str := fmt.Sprint(testVal)
		strOut := in.String()
		if strOut != str {
			t.Errorf("#%d.String got %s want %s", i, strOut, str)
		}
	}
}

var bitwiseTests = []struct {
	x, y                 string
	and, or, xor, andNot string
}{
	{"0x00", "0x00", "0x00", "0x00", "0x00", "0x00"},
	{"0x00", "0x01", "0x00", "0x01", "0x01", "0x00"},
	{"0x01", "0x00", "0x00", "0x01", "0x01", "0x01"},
	{"-0x01", "0x00", "0x00", "-0x01", "-0x01", "-0x01"},
	{"-0xaf", "-0x50", "-0xf0", "-0x0f", "0xe1", "0x41"},
	{"0x00", "-0x01", "0x00", "-0x01", "-0x01", "0x00"},
	{"0x01", "0x01", "0x01", "0x01", "0x00", "0x00"},
	{"-0x01", "-0x01", "-0x01", "-0x01", "0x00", "0x00"},
	{"0x07", "0x08", "0x00", "0x0f", "0x0f", "0x07"},
	{"0x05", "0x0f", "0x05", "0x0f", "0x0a", "0x00"},
	{"0x013ff6", "0x9a4e", "0x1a46", "0x01bffe", "0x01a5b8", "0x0125b0"},
	{"-0x013ff6", "0x9a4e", "0x800a", "-0x0125b2", "-0x01a5bc", "-0x01c000"},
	{"-0x013ff6", "-0x9a4e", "-0x01bffe", "-0x1a46", "0x01a5b8", "0x8008"},
	{
		"0x1000009dc6e3d9822cba04129bcbe3401",
		"0xb9bd7d543685789d57cb918e833af352559021483cdb05cc21fd",
		"0x1000001186210100001000009048c2001",
		"0xb9bd7d543685789d57cb918e8bfeff7fddb2ebe87dfbbdfe35fd",
		"0xb9bd7d543685789d57ca918e8ae69d6fcdb2eae87df2b97215fc",
		"0x8c40c2d8822caa04120b8321400",
	},
	{
		"0x1000009dc6e3d9822cba04129bcbe3401",
		"-0xb9bd7d543685789d57cb918e833af352559021483cdb05cc21fd",
		"0x8c40c2d8822caa04120b8321401",
		"-0xb9bd7d543685789d57ca918e82229142459020483cd2014001fd",
		"-0xb9bd7d543685789d57ca918e8ae69d6fcdb2eae87df2b97215fe",
		"0x1000001186210100001000009048c2000",
	},
	{
		"-0x1000009dc6e3d9822cba04129bcbe3401",
		"-0xb9bd7d543685789d57cb918e833af352559021483cdb05cc21fd",
		"-0xb9bd7d543685789d57cb918e8bfeff7fddb2ebe87dfbbdfe35fd",
		"-0x1000001186210100001000009048c2001",
		"0xb9bd7d543685789d57ca918e8ae69d6fcdb2eae87df2b97215fc",
		"0xb9bd7d543685789d57ca918e82229142459020483cd2014001fc",
	},
}

type bitFun func(z, x, y *Int) *Int

func testBitFun(t *testing.T, msg string, f bitFun, x, y *Int, exp string) {
	expected := new(Int)
	expected.SetString(exp, 0)

	out := f(new(Int), x, y)
	if out.Cmp(expected) != 0 {
		t.Errorf("%s: got %s want %s", msg, out, expected)
	}
}

func testBitFunSelf(t *testing.T, msg string, f bitFun, x, y *Int, exp string) {
	self := new(Int)
	self.Set(x)
	expected := new(Int)
	expected.SetString(exp, 0)

	self = f(self, self, y)
	if self.Cmp(expected) != 0 {
		t.Errorf("%s: got %s want %s", msg, self, expected)
	}
}

func altBit(x *Int, i int) uint {
	z := new(Int).Rsh(x, uint(i))
	z = z.And(z, NewInt(1))
	if z.Cmp(new(Int)) != 0 {
		return 1
	}
	return 0
}

func altSetBit(z *Int, x *Int, i int, b uint) *Int {
	one := NewInt(1)
	m := one.Lsh(one, uint(i))
	switch b {
	case 1:
		return z.Or(x, m)
	case 0:
		return z.AndNot(x, m)
	}
	panic("set bit is not 0 or 1")
}

func testBitset(t *testing.T, x *Int) {
	n := x.BitLen()
	z := new(Int).Set(x)
	z1 := new(Int).Set(x)
	for i := 0; i < n+10; i++ {
		old := z.Bit(i)
		old1 := altBit(z1, i)
		if old != old1 {
			t.Errorf("bitset: inconsistent value for Bit(%s, %d), got %v want %v", z1, i, old, old1)
		}
		z := new(Int).SetBit(z, i, 1)
		z1 := altSetBit(new(Int), z1, i, 1)
		if z.Bit(i) == 0 {
			t.Errorf("bitset: bit %d of %s got 0 want 1", i, x)
		}
		if z.Cmp(z1) != 0 {
			t.Errorf("bitset: inconsistent value after SetBit 1, got %s want %s", z, z1)
		}
		z.SetBit(z, i, 0)
		altSetBit(z1, z1, i, 0)
		if z.Bit(i) != 0 {
			t.Errorf("bitset: bit %d of %s got 1 want 0", i, x)
		}
		if z.Cmp(z1) != 0 {
			t.Errorf("bitset: inconsistent value after SetBit 0, got %s want %s", z, z1)
		}
		altSetBit(z1, z1, i, old)
		z.SetBit(z, i, old)
		if z.Cmp(z1) != 0 {
			t.Errorf("bitset: inconsistent value after SetBit old, got %s want %s", z, z1)
		}
	}
	if z.Cmp(x) != 0 {
		t.Errorf("bitset: got %s want %s", z, x)
	}
}

var bitsetTests = []struct {
	x string
	i int
	b uint
}{
	{"0", 0, 0},
	{"0", 200, 0},
	{"1", 0, 1},
	{"1", 1, 0},
	{"-1", 0, 1},
	{"-1", 200, 1},
	{"0x2000000000000000000000000000", 108, 0},
	{"0x2000000000000000000000000000", 109, 1},
	{"0x2000000000000000000000000000", 110, 0},
	{"-0x2000000000000000000000000001", 108, 1},
	{"-0x2000000000000000000000000001", 109, 0},
	{"-0x2000000000000000000000000001", 110, 1},
}

func TestBitSet(t *testing.T) {
	for _, test := range bitwiseTests {
		x := new(Int)
		x.SetString(test.x, 0)
		testBitset(t, x)
		x = new(Int)
		x.SetString(test.y, 0)
		testBitset(t, x)
	}
	for i, test := range bitsetTests {
		x := new(Int)
		x.SetString(test.x, 0)
		b := x.Bit(test.i)
		if b != test.b {
			t.Errorf("#%d got %v want %v", i, b, test.b)
		}
	}
	z := NewInt(1)
	z.SetBit(NewInt(0), 2, 1)
	if z.Cmp(NewInt(4)) != 0 {
		t.Errorf("destination leaked into result; got %s want 4", z)
	}
}

func BenchmarkBitset(b *testing.B) {
	z := new(Int)
	z.SetBit(z, 512, 1)
	b.ResetTimer()
	b.StartTimer()
	for i := b.N - 1; i >= 0; i-- {
		z.SetBit(z, i&512, 1)
	}
}

func BenchmarkBitsetNeg(b *testing.B) {
	z := NewInt(-1)
	z.SetBit(z, 512, 0)
	b.ResetTimer()
	b.StartTimer()
	for i := b.N - 1; i >= 0; i-- {
		z.SetBit(z, i&512, 0)
	}
}

func BenchmarkBitsetOrig(b *testing.B) {
	z := new(Int)
	altSetBit(z, z, 512, 1)
	b.ResetTimer()
	b.StartTimer()
	for i := b.N - 1; i >= 0; i-- {
		altSetBit(z, z, i&512, 1)
	}
}

func BenchmarkBitsetNegOrig(b *testing.B) {
	z := NewInt(-1)
	altSetBit(z, z, 512, 0)
	b.ResetTimer()
	b.StartTimer()
	for i := b.N - 1; i >= 0; i-- {
		altSetBit(z, z, i&512, 0)
	}
}

func TestBitwise(t *testing.T) {
	x := new(Int)
	y := new(Int)
	for _, test := range bitwiseTests {
		x.SetString(test.x, 0)
		y.SetString(test.y, 0)

		testBitFun(t, "and", (*Int).And, x, y, test.and)
		testBitFunSelf(t, "and", (*Int).And, x, y, test.and)
		testBitFun(t, "andNot", (*Int).AndNot, x, y, test.andNot)
		testBitFunSelf(t, "andNot", (*Int).AndNot, x, y, test.andNot)
		testBitFun(t, "or", (*Int).Or, x, y, test.or)
		testBitFunSelf(t, "or", (*Int).Or, x, y, test.or)
		testBitFun(t, "xor", (*Int).Xor, x, y, test.xor)
		testBitFunSelf(t, "xor", (*Int).Xor, x, y, test.xor)
	}
}

var notTests = []struct {
	in  string
	out string
}{
	{"0", "-1"},
	{"1", "-2"},
	{"7", "-8"},
	{"0", "-1"},
	{"-81910", "81909"},
	{
		"298472983472983471903246121093472394872319615612417471234712061",
		"-298472983472983471903246121093472394872319615612417471234712062",
	},
}

func TestNot(t *testing.T) {
	in := new(Int)
	out := new(Int)
	expected := new(Int)
	for i, test := range notTests {
		in.SetString(test.in, 10)
		expected.SetString(test.out, 10)
		out = out.Not(in)
		if out.Cmp(expected) != 0 {
			t.Errorf("#%d: got %s want %s", i, out, expected)
		}
		out = out.Not(out)
		if out.Cmp(in) != 0 {
			t.Errorf("#%d: got %s want %s", i, out, in)
		}
	}
}

var modInverseTests = []struct {
	element string
	prime   string
}{
	{"1", "7"},
	{"1", "13"},
	{"239487239847", "2410312426921032588552076022197566074856950548502459942654116941958108831682612228890093858261341614673227141477904012196503648957050582631942730706805009223062734745341073406696246014589361659774041027169249453200378729434170325843778659198143763193776859869524088940195577346119843545301547043747207749969763750084308926339295559968882457872412993810129130294592999947926365264059284647209730384947211681434464714438488520940127459844288859336526896320919633919"},
}

func TestModInverse(t *testing.T) {
	var element, prime Int
	one := NewInt(1)
	for i, test := range modInverseTests {
		(&element).SetString(test.element, 10)
		(&prime).SetString(test.prime, 10)
		inverse := new(Int).ModInverse(&element, &prime)
		inverse.Mul(inverse, &element)
		inverse.Mod(inverse, &prime)
		if inverse.Cmp(one) != 0 {
			t.Errorf("#%d: failed (e·e^(-1)=%s)", i, inverse)
		}
	}
}

var encodingTests = []string{
	"-539345864568634858364538753846587364875430589374589",
	"-678645873",
	"-100",
	"-2",
	"-1",
	"0",
	"1",
	"2",
	"10",
	"42",
	"1234567890",
	"298472983472983471903246121093472394872319615612417471234712061",
}

func TestIntGobEncoding(t *testing.T) {
	var medium bytes.Buffer
	enc := gob.NewEncoder(&medium)
	dec := gob.NewDecoder(&medium)
	for _, test := range encodingTests {
		medium.Reset() // empty buffer for each test case (in case of failures)
		var tx Int
		tx.SetString(test, 10)
		if err := enc.Encode(&tx); err != nil {
			t.Errorf("encoding of %s failed: %s", &tx, err)
		}
		var rx Int
		if err := dec.Decode(&rx); err != nil {
			t.Errorf("decoding of %s failed: %s", &tx, err)
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("transmission of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

// Sending a nil Int pointer (inside a slice) on a round trip through gob should yield a zero.
// TODO: top-level nils.
func TestGobEncodingNilIntInSlice(t *testing.T) {
	buf := new(bytes.Buffer)
	enc := gob.NewEncoder(buf)
	dec := gob.NewDecoder(buf)

	var in = make([]*Int, 1)
	err := enc.Encode(&in)
	if err != nil {
		t.Errorf("gob encode failed: %q", err)
	}
	var out []*Int
	err = dec.Decode(&out)
	if err != nil {
		t.Fatalf("gob decode failed: %q", err)
	}
	if len(out) != 1 {
		t.Fatalf("wrong len; want 1 got %d", len(out))
	}
	var zero Int
	if out[0].Cmp(&zero) != 0 {
		t.Errorf("transmission of (*Int)(nill) failed: got %s want 0", out)
	}
}

func TestIntJSONEncoding(t *testing.T) {
	for _, test := range encodingTests {
		var tx Int
		tx.SetString(test, 10)
		b, err := json.Marshal(&tx)
		if err != nil {
			t.Errorf("marshaling of %s failed: %s", &tx, err)
		}
		var rx Int
		if err := json.Unmarshal(b, &rx); err != nil {
			t.Errorf("unmarshaling of %s failed: %s", &tx, err)
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("JSON encoding of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

var intVals = []string{
	"-141592653589793238462643383279502884197169399375105820974944592307816406286",
	"-1415926535897932384626433832795028841971",
	"-141592653589793",
	"-1",
	"0",
	"1",
	"141592653589793",
	"1415926535897932384626433832795028841971",
	"141592653589793238462643383279502884197169399375105820974944592307816406286",
}

func TestIntJSONEncodingTextMarshaller(t *testing.T) {
	for _, num := range intVals {
		var tx Int
		tx.SetString(num, 0)
		b, err := json.Marshal(&tx)
		if err != nil {
			t.Errorf("marshaling of %s failed: %s", &tx, err)
			continue
		}
		var rx Int
		if err := json.Unmarshal(b, &rx); err != nil {
			t.Errorf("unmarshaling of %s failed: %s", &tx, err)
			continue
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("JSON encoding of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

func TestIntXMLEncodingTextMarshaller(t *testing.T) {
	for _, num := range intVals {
		var tx Int
		tx.SetString(num, 0)
		b, err := xml.Marshal(&tx)
		if err != nil {
			t.Errorf("marshaling of %s failed: %s", &tx, err)
			continue
		}
		var rx Int
		if err := xml.Unmarshal(b, &rx); err != nil {
			t.Errorf("unmarshaling of %s failed: %s", &tx, err)
			continue
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("XML encoding of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

func TestIssue2607(t *testing.T) {
	// This code sequence used to hang.
	n := NewInt(10)
	n.Rand(rand.New(rand.NewSource(9)), n)
}
