// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"fmt"
	"gob"
	"testing"
)

func TestZeroRat(t *testing.T) {
	var x, y, z Rat
	y.SetFrac64(0, 42)

	if x.Cmp(&y) != 0 {
		t.Errorf("x and y should be both equal and zero")
	}

	if s := x.String(); s != "0/1" {
		t.Errorf("got x = %s, want 0/1", s)
	}

	if s := x.RatString(); s != "0" {
		t.Errorf("got x = %s, want 0", s)
	}

	z.Add(&x, &y)
	if s := z.RatString(); s != "0" {
		t.Errorf("got x+y = %s, want 0", s)
	}

	z.Sub(&x, &y)
	if s := z.RatString(); s != "0" {
		t.Errorf("got x-y = %s, want 0", s)
	}

	z.Mul(&x, &y)
	if s := z.RatString(); s != "0" {
		t.Errorf("got x*y = %s, want 0", s)
	}

	// check for division by zero
	defer func() {
		if s := recover(); s == nil || s.(string) != "division by zero" {
			panic(s)
		}
	}()
	z.Quo(&x, &y)
}

var setStringTests = []struct {
	in, out string
	ok      bool
}{
	{"0", "0", true},
	{"-0", "0", true},
	{"1", "1", true},
	{"-1", "-1", true},
	{"1.", "1", true},
	{"1e0", "1", true},
	{"1.e1", "10", true},
	{in: "1e", ok: false},
	{in: "1.e", ok: false},
	{in: "1e+14e-5", ok: false},
	{in: "1e4.5", ok: false},
	{in: "r", ok: false},
	{in: "a/b", ok: false},
	{in: "a.b", ok: false},
	{"-0.1", "-1/10", true},
	{"-.1", "-1/10", true},
	{"2/4", "1/2", true},
	{".25", "1/4", true},
	{"-1/5", "-1/5", true},
	{"8129567.7690E14", "812956776900000000000", true},
	{"78189e+4", "781890000", true},
	{"553019.8935e+8", "55301989350000", true},
	{"98765432109876543210987654321e-10", "98765432109876543210987654321/10000000000", true},
	{"9877861857500000E-7", "3951144743/4", true},
	{"2169378.417e-3", "2169378417/1000000", true},
	{"884243222337379604041632732738665534", "884243222337379604041632732738665534", true},
	{"53/70893980658822810696", "53/70893980658822810696", true},
	{"106/141787961317645621392", "53/70893980658822810696", true},
	{"204211327800791583.81095", "4084226556015831676219/20000", true},
}

func TestRatSetString(t *testing.T) {
	for i, test := range setStringTests {
		x, ok := new(Rat).SetString(test.in)

		if ok {
			if !test.ok {
				t.Errorf("#%d SetString(%q) expected failure", i, test.in)
			} else if x.RatString() != test.out {
				t.Errorf("#%d SetString(%q) got %s want %s", i, test.in, x.RatString(), test.out)
			}
		} else if x != nil {
			t.Errorf("#%d SetString(%q) got %p want nil", i, test.in, x)
		}
	}
}

func TestRatScan(t *testing.T) {
	var buf bytes.Buffer
	for i, test := range setStringTests {
		x := new(Rat)
		buf.Reset()
		buf.WriteString(test.in)

		_, err := fmt.Fscanf(&buf, "%v", x)
		if err == nil != test.ok {
			if test.ok {
				t.Errorf("#%d error: %s", i, err)
			} else {
				t.Errorf("#%d expected error", i)
			}
			continue
		}
		if err == nil && x.RatString() != test.out {
			t.Errorf("#%d got %s want %s", i, x.RatString(), test.out)
		}
	}
}

var floatStringTests = []struct {
	in   string
	prec int
	out  string
}{
	{"0", 0, "0"},
	{"0", 4, "0.0000"},
	{"1", 0, "1"},
	{"1", 2, "1.00"},
	{"-1", 0, "-1"},
	{".25", 2, "0.25"},
	{".25", 1, "0.3"},
	{".25", 3, "0.250"},
	{"-1/3", 3, "-0.333"},
	{"-2/3", 4, "-0.6667"},
	{"0.96", 1, "1.0"},
	{"0.999", 2, "1.00"},
	{"0.9", 0, "1"},
	{".25", -1, "0"},
	{".55", -1, "1"},
}

func TestFloatString(t *testing.T) {
	for i, test := range floatStringTests {
		x, _ := new(Rat).SetString(test.in)

		if x.FloatString(test.prec) != test.out {
			t.Errorf("#%d got %s want %s", i, x.FloatString(test.prec), test.out)
		}
	}
}

func TestRatSign(t *testing.T) {
	zero := NewRat(0, 1)
	for _, a := range setStringTests {
		x, ok := new(Rat).SetString(a.in)
		if !ok {
			continue
		}
		s := x.Sign()
		e := x.Cmp(zero)
		if s != e {
			t.Errorf("got %d; want %d for z = %v", s, e, &x)
		}
	}
}

var ratCmpTests = []struct {
	rat1, rat2 string
	out        int
}{
	{"0", "0/1", 0},
	{"1/1", "1", 0},
	{"-1", "-2/2", 0},
	{"1", "0", 1},
	{"0/1", "1/1", -1},
	{"-5/1434770811533343057144", "-5/1434770811533343057145", -1},
	{"49832350382626108453/8964749413", "49832350382626108454/8964749413", -1},
	{"-37414950961700930/7204075375675961", "37414950961700930/7204075375675961", -1},
	{"37414950961700930/7204075375675961", "74829901923401860/14408150751351922", 0},
}

func TestRatCmp(t *testing.T) {
	for i, test := range ratCmpTests {
		x, _ := new(Rat).SetString(test.rat1)
		y, _ := new(Rat).SetString(test.rat2)

		out := x.Cmp(y)
		if out != test.out {
			t.Errorf("#%d got out = %v; want %v", i, out, test.out)
		}
	}
}

func TestIsInt(t *testing.T) {
	one := NewInt(1)
	for _, a := range setStringTests {
		x, ok := new(Rat).SetString(a.in)
		if !ok {
			continue
		}
		i := x.IsInt()
		e := x.Denom().Cmp(one) == 0
		if i != e {
			t.Errorf("got IsInt(%v) == %v; want %v", x, i, e)
		}
	}
}

func TestRatAbs(t *testing.T) {
	zero := new(Rat)
	for _, a := range setStringTests {
		x, ok := new(Rat).SetString(a.in)
		if !ok {
			continue
		}
		e := new(Rat).Set(x)
		if e.Cmp(zero) < 0 {
			e.Sub(zero, e)
		}
		z := new(Rat).Abs(x)
		if z.Cmp(e) != 0 {
			t.Errorf("got Abs(%v) = %v; want %v", x, z, e)
		}
	}
}

func TestRatNeg(t *testing.T) {
	zero := new(Rat)
	for _, a := range setStringTests {
		x, ok := new(Rat).SetString(a.in)
		if !ok {
			continue
		}
		e := new(Rat).Sub(zero, x)
		z := new(Rat).Neg(x)
		if z.Cmp(e) != 0 {
			t.Errorf("got Neg(%v) = %v; want %v", x, z, e)
		}
	}
}

func TestRatInv(t *testing.T) {
	zero := new(Rat)
	for _, a := range setStringTests {
		x, ok := new(Rat).SetString(a.in)
		if !ok {
			continue
		}
		if x.Cmp(zero) == 0 {
			continue // avoid division by zero
		}
		e := new(Rat).SetFrac(x.Denom(), x.Num())
		z := new(Rat).Inv(x)
		if z.Cmp(e) != 0 {
			t.Errorf("got Inv(%v) = %v; want %v", x, z, e)
		}
	}
}

type ratBinFun func(z, x, y *Rat) *Rat
type ratBinArg struct {
	x, y, z string
}

func testRatBin(t *testing.T, i int, name string, f ratBinFun, a ratBinArg) {
	x, _ := new(Rat).SetString(a.x)
	y, _ := new(Rat).SetString(a.y)
	z, _ := new(Rat).SetString(a.z)
	out := f(new(Rat), x, y)

	if out.Cmp(z) != 0 {
		t.Errorf("%s #%d got %s want %s", name, i, out, z)
	}
}

var ratBinTests = []struct {
	x, y      string
	sum, prod string
}{
	{"0", "0", "0", "0"},
	{"0", "1", "1", "0"},
	{"-1", "0", "-1", "0"},
	{"-1", "1", "0", "-1"},
	{"1", "1", "2", "1"},
	{"1/2", "1/2", "1", "1/4"},
	{"1/4", "1/3", "7/12", "1/12"},
	{"2/5", "-14/3", "-64/15", "-28/15"},
	{"4707/49292519774798173060", "-3367/70976135186689855734", "84058377121001851123459/1749296273614329067191168098769082663020", "-1760941/388732505247628681598037355282018369560"},
	{"-61204110018146728334/3", "-31052192278051565633/2", "-215564796870448153567/6", "950260896245257153059642991192710872711/3"},
	{"-854857841473707320655/4237645934602118692642972629634714039", "-18/31750379913563777419", "-27/133467566250814981", "15387441146526731771790/134546868362786310073779084329032722548987800600710485341"},
	{"618575745270541348005638912139/19198433543745179392300736", "-19948846211000086/637313996471", "27674141753240653/30123979153216", "-6169936206128396568797607742807090270137721977/6117715203873571641674006593837351328"},
	{"-3/26206484091896184128", "5/2848423294177090248", "15310893822118706237/9330894968229805033368778458685147968", "-5/24882386581946146755650075889827061248"},
	{"26946729/330400702820", "41563965/225583428284", "1238218672302860271/4658307703098666660055", "224002580204097/14906584649915733312176"},
	{"-8259900599013409474/7", "-84829337473700364773/56707961321161574960", "-468402123685491748914621885145127724451/396955729248131024720", "350340947706464153265156004876107029701/198477864624065512360"},
	{"575775209696864/1320203974639986246357", "29/712593081308", "410331716733912717985762465/940768218243776489278275419794956", "808/45524274987585732633"},
	{"1786597389946320496771/2066653520653241", "6269770/1992362624741777", "3559549865190272133656109052308126637/4117523232840525481453983149257", "8967230/3296219033"},
	{"-36459180403360509753/32150500941194292113930", "9381566963714/9633539", "301622077145533298008420642898530153/309723104686531919656937098270", "-3784609207827/3426986245"},
}

func TestRatBin(t *testing.T) {
	for i, test := range ratBinTests {
		arg := ratBinArg{test.x, test.y, test.sum}
		testRatBin(t, i, "Add", (*Rat).Add, arg)

		arg = ratBinArg{test.y, test.x, test.sum}
		testRatBin(t, i, "Add symmetric", (*Rat).Add, arg)

		arg = ratBinArg{test.sum, test.x, test.y}
		testRatBin(t, i, "Sub", (*Rat).Sub, arg)

		arg = ratBinArg{test.sum, test.y, test.x}
		testRatBin(t, i, "Sub symmetric", (*Rat).Sub, arg)

		arg = ratBinArg{test.x, test.y, test.prod}
		testRatBin(t, i, "Mul", (*Rat).Mul, arg)

		arg = ratBinArg{test.y, test.x, test.prod}
		testRatBin(t, i, "Mul symmetric", (*Rat).Mul, arg)

		if test.x != "0" {
			arg = ratBinArg{test.prod, test.x, test.y}
			testRatBin(t, i, "Quo", (*Rat).Quo, arg)
		}

		if test.y != "0" {
			arg = ratBinArg{test.prod, test.y, test.x}
			testRatBin(t, i, "Quo symmetric", (*Rat).Quo, arg)
		}
	}
}

func TestIssue820(t *testing.T) {
	x := NewRat(3, 1)
	y := NewRat(2, 1)
	z := y.Quo(x, y)
	q := NewRat(3, 2)
	if z.Cmp(q) != 0 {
		t.Errorf("got %s want %s", z, q)
	}

	y = NewRat(3, 1)
	x = NewRat(2, 1)
	z = y.Quo(x, y)
	q = NewRat(2, 3)
	if z.Cmp(q) != 0 {
		t.Errorf("got %s want %s", z, q)
	}

	x = NewRat(3, 1)
	z = x.Quo(x, x)
	q = NewRat(3, 3)
	if z.Cmp(q) != 0 {
		t.Errorf("got %s want %s", z, q)
	}
}

var setFrac64Tests = []struct {
	a, b int64
	out  string
}{
	{0, 1, "0"},
	{0, -1, "0"},
	{1, 1, "1"},
	{-1, 1, "-1"},
	{1, -1, "-1"},
	{-1, -1, "1"},
	{-9223372036854775808, -9223372036854775808, "1"},
}

func TestRatSetFrac64Rat(t *testing.T) {
	for i, test := range setFrac64Tests {
		x := new(Rat).SetFrac64(test.a, test.b)
		if x.RatString() != test.out {
			t.Errorf("#%d got %s want %s", i, x.RatString(), test.out)
		}
	}
}

func TestRatGobEncoding(t *testing.T) {
	var medium bytes.Buffer
	enc := gob.NewEncoder(&medium)
	dec := gob.NewDecoder(&medium)
	for i, test := range gobEncodingTests {
		for j := 0; j < 4; j++ {
			medium.Reset() // empty buffer for each test case (in case of failures)
			stest := test
			if j&1 != 0 {
				// negative numbers
				stest = "-" + test
			}
			if j%2 != 0 {
				// fractions
				stest = stest + "." + test
			}
			var tx Rat
			tx.SetString(stest)
			if err := enc.Encode(&tx); err != nil {
				t.Errorf("#%d%c: encoding failed: %s", i, 'a'+j, err)
			}
			var rx Rat
			if err := dec.Decode(&rx); err != nil {
				t.Errorf("#%d%c: decoding failed: %s", i, 'a'+j, err)
			}
			if rx.Cmp(&tx) != 0 {
				t.Errorf("#%d%c: transmission failed: got %s want %s", i, 'a'+j, &rx, &tx)
			}
		}
	}
}

func TestIssue2379(t *testing.T) {
	// 1) no aliasing
	q := NewRat(3, 2)
	x := new(Rat)
	x.SetFrac(NewInt(3), NewInt(2))
	if x.Cmp(q) != 0 {
		t.Errorf("1) got %s want %s", x, q)
	}

	// 2) aliasing of numerator
	x = NewRat(2, 3)
	x.SetFrac(NewInt(3), x.Num())
	if x.Cmp(q) != 0 {
		t.Errorf("2) got %s want %s", x, q)
	}

	// 3) aliasing of denominator
	x = NewRat(2, 3)
	x.SetFrac(x.Denom(), NewInt(2))
	if x.Cmp(q) != 0 {
		t.Errorf("3) got %s want %s", x, q)
	}

	// 4) aliasing of numerator and denominator
	x = NewRat(2, 3)
	x.SetFrac(x.Denom(), x.Num())
	if x.Cmp(q) != 0 {
		t.Errorf("4) got %s want %s", x, q)
	}

	// 5) numerator and denominator are the same
	q = NewRat(1, 1)
	x = new(Rat)
	n := NewInt(7)
	x.SetFrac(n, n)
	if x.Cmp(q) != 0 {
		t.Errorf("5) got %s want %s", x, q)
	}
}
