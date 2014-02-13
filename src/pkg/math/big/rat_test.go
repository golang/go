// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"math"
	"strconv"
	"strings"
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
	for _, test := range encodingTests {
		medium.Reset() // empty buffer for each test case (in case of failures)
		var tx Rat
		tx.SetString(test + ".14159265")
		if err := enc.Encode(&tx); err != nil {
			t.Errorf("encoding of %s failed: %s", &tx, err)
		}
		var rx Rat
		if err := dec.Decode(&rx); err != nil {
			t.Errorf("decoding of %s failed: %s", &tx, err)
		}
		if rx.Cmp(&tx) != 0 {
			t.Errorf("transmission of %s failed: got %s want %s", &tx, &rx, &tx)
		}
	}
}

// Sending a nil Rat pointer (inside a slice) on a round trip through gob should yield a zero.
// TODO: top-level nils.
func TestGobEncodingNilRatInSlice(t *testing.T) {
	buf := new(bytes.Buffer)
	enc := gob.NewEncoder(buf)
	dec := gob.NewDecoder(buf)

	var in = make([]*Rat, 1)
	err := enc.Encode(&in)
	if err != nil {
		t.Errorf("gob encode failed: %q", err)
	}
	var out []*Rat
	err = dec.Decode(&out)
	if err != nil {
		t.Fatalf("gob decode failed: %q", err)
	}
	if len(out) != 1 {
		t.Fatalf("wrong len; want 1 got %d", len(out))
	}
	var zero Rat
	if out[0].Cmp(&zero) != 0 {
		t.Errorf("transmission of (*Int)(nill) failed: got %s want 0", out)
	}
}

var ratNums = []string{
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

var ratDenoms = []string{
	"1",
	"718281828459045",
	"7182818284590452353602874713526624977572",
	"718281828459045235360287471352662497757247093699959574966967627724076630353",
}

func TestRatJSONEncoding(t *testing.T) {
	for _, num := range ratNums {
		for _, denom := range ratDenoms {
			var tx Rat
			tx.SetString(num + "/" + denom)
			b, err := json.Marshal(&tx)
			if err != nil {
				t.Errorf("marshaling of %s failed: %s", &tx, err)
				continue
			}
			var rx Rat
			if err := json.Unmarshal(b, &rx); err != nil {
				t.Errorf("unmarshaling of %s failed: %s", &tx, err)
				continue
			}
			if rx.Cmp(&tx) != 0 {
				t.Errorf("JSON encoding of %s failed: got %s want %s", &tx, &rx, &tx)
			}
		}
	}
}

func TestRatXMLEncoding(t *testing.T) {
	for _, num := range ratNums {
		for _, denom := range ratDenoms {
			var tx Rat
			tx.SetString(num + "/" + denom)
			b, err := xml.Marshal(&tx)
			if err != nil {
				t.Errorf("marshaling of %s failed: %s", &tx, err)
				continue
			}
			var rx Rat
			if err := xml.Unmarshal(b, &rx); err != nil {
				t.Errorf("unmarshaling of %s failed: %s", &tx, err)
				continue
			}
			if rx.Cmp(&tx) != 0 {
				t.Errorf("XML encoding of %s failed: got %s want %s", &tx, &rx, &tx)
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

func TestIssue3521(t *testing.T) {
	a := new(Int)
	b := new(Int)
	a.SetString("64375784358435883458348587", 0)
	b.SetString("4789759874531", 0)

	// 0) a raw zero value has 1 as denominator
	zero := new(Rat)
	one := NewInt(1)
	if zero.Denom().Cmp(one) != 0 {
		t.Errorf("0) got %s want %s", zero.Denom(), one)
	}

	// 1a) a zero value remains zero independent of denominator
	x := new(Rat)
	x.Denom().Set(new(Int).Neg(b))
	if x.Cmp(zero) != 0 {
		t.Errorf("1a) got %s want %s", x, zero)
	}

	// 1b) a zero value may have a denominator != 0 and != 1
	x.Num().Set(a)
	qab := new(Rat).SetFrac(a, b)
	if x.Cmp(qab) != 0 {
		t.Errorf("1b) got %s want %s", x, qab)
	}

	// 2a) an integral value becomes a fraction depending on denominator
	x.SetFrac64(10, 2)
	x.Denom().SetInt64(3)
	q53 := NewRat(5, 3)
	if x.Cmp(q53) != 0 {
		t.Errorf("2a) got %s want %s", x, q53)
	}

	// 2b) an integral value becomes a fraction depending on denominator
	x = NewRat(10, 2)
	x.Denom().SetInt64(3)
	if x.Cmp(q53) != 0 {
		t.Errorf("2b) got %s want %s", x, q53)
	}

	// 3) changing the numerator/denominator of a Rat changes the Rat
	x.SetFrac(a, b)
	a = x.Num()
	b = x.Denom()
	a.SetInt64(5)
	b.SetInt64(3)
	if x.Cmp(q53) != 0 {
		t.Errorf("3) got %s want %s", x, q53)
	}
}

// Test inputs to Rat.SetString.  The prefix "long:" causes the test
// to be skipped in --test.short mode.  (The threshold is about 500us.)
var float64inputs = []string{
	// Constants plundered from strconv/testfp.txt.

	// Table 1: Stress Inputs for Conversion to 53-bit Binary, < 1/2 ULP
	"5e+125",
	"69e+267",
	"999e-026",
	"7861e-034",
	"75569e-254",
	"928609e-261",
	"9210917e+080",
	"84863171e+114",
	"653777767e+273",
	"5232604057e-298",
	"27235667517e-109",
	"653532977297e-123",
	"3142213164987e-294",
	"46202199371337e-072",
	"231010996856685e-073",
	"9324754620109615e+212",
	"78459735791271921e+049",
	"272104041512242479e+200",
	"6802601037806061975e+198",
	"20505426358836677347e-221",
	"836168422905420598437e-234",
	"4891559871276714924261e+222",

	// Table 2: Stress Inputs for Conversion to 53-bit Binary, > 1/2 ULP
	"9e-265",
	"85e-037",
	"623e+100",
	"3571e+263",
	"81661e+153",
	"920657e-023",
	"4603285e-024",
	"87575437e-309",
	"245540327e+122",
	"6138508175e+120",
	"83356057653e+193",
	"619534293513e+124",
	"2335141086879e+218",
	"36167929443327e-159",
	"609610927149051e-255",
	"3743626360493413e-165",
	"94080055902682397e-242",
	"899810892172646163e+283",
	"7120190517612959703e+120",
	"25188282901709339043e-252",
	"308984926168550152811e-052",
	"6372891218502368041059e+064",

	// Table 14: Stress Inputs for Conversion to 24-bit Binary, <1/2 ULP
	"5e-20",
	"67e+14",
	"985e+15",
	"7693e-42",
	"55895e-16",
	"996622e-44",
	"7038531e-32",
	"60419369e-46",
	"702990899e-20",
	"6930161142e-48",
	"25933168707e+13",
	"596428896559e+20",

	// Table 15: Stress Inputs for Conversion to 24-bit Binary, >1/2 ULP
	"3e-23",
	"57e+18",
	"789e-35",
	"2539e-18",
	"76173e+28",
	"887745e-11",
	"5382571e-37",
	"82381273e-35",
	"750486563e-38",
	"3752432815e-39",
	"75224575729e-45",
	"459926601011e+15",

	// Constants plundered from strconv/atof_test.go.

	"0",
	"1",
	"+1",
	"1e23",
	"1E23",
	"100000000000000000000000",
	"1e-100",
	"123456700",
	"99999999999999974834176",
	"100000000000000000000001",
	"100000000000000008388608",
	"100000000000000016777215",
	"100000000000000016777216",
	"-1",
	"-0.1",
	"-0", // NB: exception made for this input
	"1e-20",
	"625e-3",

	// largest float64
	"1.7976931348623157e308",
	"-1.7976931348623157e308",
	// next float64 - too large
	"1.7976931348623159e308",
	"-1.7976931348623159e308",
	// the border is ...158079
	// borderline - okay
	"1.7976931348623158e308",
	"-1.7976931348623158e308",
	// borderline - too large
	"1.797693134862315808e308",
	"-1.797693134862315808e308",

	// a little too large
	"1e308",
	"2e308",
	"1e309",

	// way too large
	"1e310",
	"-1e310",
	"1e400",
	"-1e400",
	"long:1e400000",
	"long:-1e400000",

	// denormalized
	"1e-305",
	"1e-306",
	"1e-307",
	"1e-308",
	"1e-309",
	"1e-310",
	"1e-322",
	// smallest denormal
	"5e-324",
	"4e-324",
	"3e-324",
	// too small
	"2e-324",
	// way too small
	"1e-350",
	"long:1e-400000",
	// way too small, negative
	"-1e-350",
	"long:-1e-400000",

	// try to overflow exponent
	// [Disabled: too slow and memory-hungry with rationals.]
	// "1e-4294967296",
	// "1e+4294967296",
	// "1e-18446744073709551616",
	// "1e+18446744073709551616",

	// http://www.exploringbinary.com/java-hangs-when-converting-2-2250738585072012e-308/
	"2.2250738585072012e-308",
	// http://www.exploringbinary.com/php-hangs-on-numeric-value-2-2250738585072011e-308/

	"2.2250738585072011e-308",

	// A very large number (initially wrongly parsed by the fast algorithm).
	"4.630813248087435e+307",

	// A different kind of very large number.
	"22.222222222222222",
	"long:2." + strings.Repeat("2", 4000) + "e+1",

	// Exactly halfway between 1 and math.Nextafter(1, 2).
	// Round to even (down).
	"1.00000000000000011102230246251565404236316680908203125",
	// Slightly lower; still round down.
	"1.00000000000000011102230246251565404236316680908203124",
	// Slightly higher; round up.
	"1.00000000000000011102230246251565404236316680908203126",
	// Slightly higher, but you have to read all the way to the end.
	"long:1.00000000000000011102230246251565404236316680908203125" + strings.Repeat("0", 10000) + "1",

	// Smallest denormal, 2^(-1022-52)
	"4.940656458412465441765687928682213723651e-324",
	// Half of smallest denormal, 2^(-1022-53)
	"2.470328229206232720882843964341106861825e-324",
	// A little more than the exact half of smallest denormal
	// 2^-1075 + 2^-1100.  (Rounds to 1p-1074.)
	"2.470328302827751011111470718709768633275e-324",
	// The exact halfway between smallest normal and largest denormal:
	// 2^-1022 - 2^-1075.  (Rounds to 2^-1022.)
	"2.225073858507201136057409796709131975935e-308",

	"1152921504606846975",  //   1<<60 - 1
	"-1152921504606846975", // -(1<<60 - 1)
	"1152921504606846977",  //   1<<60 + 1
	"-1152921504606846977", // -(1<<60 + 1)

	"1/3",
}

func TestFloat64SpecialCases(t *testing.T) {
	for _, input := range float64inputs {
		if strings.HasPrefix(input, "long:") {
			if testing.Short() {
				continue
			}
			input = input[len("long:"):]
		}

		r, ok := new(Rat).SetString(input)
		if !ok {
			t.Errorf("Rat.SetString(%q) failed", input)
			continue
		}
		f, exact := r.Float64()

		// 1. Check string -> Rat -> float64 conversions are
		// consistent with strconv.ParseFloat.
		// Skip this check if the input uses "a/b" rational syntax.
		if !strings.Contains(input, "/") {
			e, _ := strconv.ParseFloat(input, 64)

			// Careful: negative Rats too small for
			// float64 become -0, but Rat obviously cannot
			// preserve the sign from SetString("-0").
			switch {
			case math.Float64bits(e) == math.Float64bits(f):
				// Ok: bitwise equal.
			case f == 0 && r.Num().BitLen() == 0:
				// Ok: Rat(0) is equivalent to both +/- float64(0).
			default:
				t.Errorf("strconv.ParseFloat(%q) = %g (%b), want %g (%b); delta = %g", input, e, e, f, f, f-e)
			}
		}

		if !isFinite(f) {
			continue
		}

		// 2. Check f is best approximation to r.
		if !checkIsBestApprox(t, f, r) {
			// Append context information.
			t.Errorf("(input was %q)", input)
		}

		// 3. Check f->R->f roundtrip is non-lossy.
		checkNonLossyRoundtrip(t, f)

		// 4. Check exactness using slow algorithm.
		if wasExact := new(Rat).SetFloat64(f).Cmp(r) == 0; wasExact != exact {
			t.Errorf("Rat.SetString(%q).Float64().exact = %t, want %t", input, exact, wasExact)
		}
	}
}

func TestFloat64Distribution(t *testing.T) {
	// Generate a distribution of (sign, mantissa, exp) values
	// broader than the float64 range, and check Rat.Float64()
	// always picks the closest float64 approximation.
	var add = []int64{
		0,
		1,
		3,
		5,
		7,
		9,
		11,
	}
	var winc, einc = uint64(1), int(1) // soak test (~75s on x86-64)
	if testing.Short() {
		winc, einc = 10, 500 // quick test (~12ms on x86-64)
	}

	for _, sign := range "+-" {
		for _, a := range add {
			for wid := uint64(0); wid < 60; wid += winc {
				b := int64(1<<wid + a)
				if sign == '-' {
					b = -b
				}
				for exp := -1100; exp < 1100; exp += einc {
					num, den := NewInt(b), NewInt(1)
					if exp > 0 {
						num.Lsh(num, uint(exp))
					} else {
						den.Lsh(den, uint(-exp))
					}
					r := new(Rat).SetFrac(num, den)
					f, _ := r.Float64()

					if !checkIsBestApprox(t, f, r) {
						// Append context information.
						t.Errorf("(input was mantissa %#x, exp %d; f = %g (%b); f ~ %g; r = %v)",
							b, exp, f, f, math.Ldexp(float64(b), exp), r)
					}

					checkNonLossyRoundtrip(t, f)
				}
			}
		}
	}
}

// TestFloat64NonFinite checks that SetFloat64 of a non-finite value
// returns nil.
func TestSetFloat64NonFinite(t *testing.T) {
	for _, f := range []float64{math.NaN(), math.Inf(+1), math.Inf(-1)} {
		var r Rat
		if r2 := r.SetFloat64(f); r2 != nil {
			t.Errorf("SetFloat64(%g) was %v, want nil", f, r2)
		}
	}
}

// checkNonLossyRoundtrip checks that a float->Rat->float roundtrip is
// non-lossy for finite f.
func checkNonLossyRoundtrip(t *testing.T, f float64) {
	if !isFinite(f) {
		return
	}
	r := new(Rat).SetFloat64(f)
	if r == nil {
		t.Errorf("Rat.SetFloat64(%g (%b)) == nil", f, f)
		return
	}
	f2, exact := r.Float64()
	if f != f2 || !exact {
		t.Errorf("Rat.SetFloat64(%g).Float64() = %g (%b), %v, want %g (%b), %v; delta = %b",
			f, f2, f2, exact, f, f, true, f2-f)
	}
}

// delta returns the absolute difference between r and f.
func delta(r *Rat, f float64) *Rat {
	d := new(Rat).Sub(r, new(Rat).SetFloat64(f))
	return d.Abs(d)
}

// checkIsBestApprox checks that f is the best possible float64
// approximation of r.
// Returns true on success.
func checkIsBestApprox(t *testing.T, f float64, r *Rat) bool {
	if math.Abs(f) >= math.MaxFloat64 {
		// Cannot check +Inf, -Inf, nor the float next to them (MaxFloat64).
		// But we have tests for these special cases.
		return true
	}

	// r must be strictly between f0 and f1, the floats bracketing f.
	f0 := math.Nextafter(f, math.Inf(-1))
	f1 := math.Nextafter(f, math.Inf(+1))

	// For f to be correct, r must be closer to f than to f0 or f1.
	df := delta(r, f)
	df0 := delta(r, f0)
	df1 := delta(r, f1)
	if df.Cmp(df0) > 0 {
		t.Errorf("Rat(%v).Float64() = %g (%b), but previous float64 %g (%b) is closer", r, f, f, f0, f0)
		return false
	}
	if df.Cmp(df1) > 0 {
		t.Errorf("Rat(%v).Float64() = %g (%b), but next float64 %g (%b) is closer", r, f, f, f1, f1)
		return false
	}
	if df.Cmp(df0) == 0 && !isEven(f) {
		t.Errorf("Rat(%v).Float64() = %g (%b); halfway should have rounded to %g (%b) instead", r, f, f, f0, f0)
		return false
	}
	if df.Cmp(df1) == 0 && !isEven(f) {
		t.Errorf("Rat(%v).Float64() = %g (%b); halfway should have rounded to %g (%b) instead", r, f, f, f1, f1)
		return false
	}
	return true
}

func isEven(f float64) bool { return math.Float64bits(f)&1 == 0 }

func TestIsFinite(t *testing.T) {
	finites := []float64{
		1.0 / 3,
		4891559871276714924261e+222,
		math.MaxFloat64,
		math.SmallestNonzeroFloat64,
		-math.MaxFloat64,
		-math.SmallestNonzeroFloat64,
	}
	for _, f := range finites {
		if !isFinite(f) {
			t.Errorf("!IsFinite(%g (%b))", f, f)
		}
	}
	nonfinites := []float64{
		math.NaN(),
		math.Inf(-1),
		math.Inf(+1),
	}
	for _, f := range nonfinites {
		if isFinite(f) {
			t.Errorf("IsFinite(%g, (%b))", f, f)
		}
	}
}
