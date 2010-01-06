// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"encoding/hex"
	"testing"
	"testing/quick"
)

func newZ(x int64) *Int {
	var z Int
	return z.New(x)
}


type funZZ func(z, x, y *Int) *Int
type argZZ struct {
	z, x, y *Int
}


var sumZZ = []argZZ{
	argZZ{newZ(0), newZ(0), newZ(0)},
	argZZ{newZ(1), newZ(1), newZ(0)},
	argZZ{newZ(1111111110), newZ(123456789), newZ(987654321)},
	argZZ{newZ(-1), newZ(-1), newZ(0)},
	argZZ{newZ(864197532), newZ(-123456789), newZ(987654321)},
	argZZ{newZ(-1111111110), newZ(-123456789), newZ(-987654321)},
}


var prodZZ = []argZZ{
	argZZ{newZ(0), newZ(0), newZ(0)},
	argZZ{newZ(0), newZ(1), newZ(0)},
	argZZ{newZ(1), newZ(1), newZ(1)},
	argZZ{newZ(-991 * 991), newZ(991), newZ(-991)},
	// TODO(gri) add larger products
}


func TestSetZ(t *testing.T) {
	for _, a := range sumZZ {
		var z Int
		z.Set(a.z)
		if (&z).Cmp(a.z) != 0 {
			t.Errorf("got z = %v; want %v", z, a.z)
		}
	}
}


func testFunZZ(t *testing.T, msg string, f funZZ, a argZZ) {
	var z Int
	f(&z, a.x, a.y)
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


var facts = map[int]string{
	0: "1",
	1: "1",
	2: "2",
	10: "3628800",
	20: "2432902008176640000",
	100: "933262154439441526816992388562667004907159682643816214685929" +
		"638952175999932299156089414639761565182862536979208272237582" +
		"51185210916864000000000000000000000000",
}


func fact(n int) *Int {
	var z Int
	z.New(1)
	for i := 2; i <= n; i++ {
		var t Int
		t.New(int64(i))
		z.Mul(&z, &t)
	}
	return &z
}


func TestFact(t *testing.T) {
	for n, s := range facts {
		f := fact(n).String()
		if f != s {
			t.Errorf("%d! = %s; want %s", n, f, s)
		}
	}
}


type fromStringTest struct {
	in   string
	base int
	out  int64
	ok   bool
}


var fromStringTests = []fromStringTest{
	fromStringTest{in: "", ok: false},
	fromStringTest{in: "a", ok: false},
	fromStringTest{in: "z", ok: false},
	fromStringTest{in: "+", ok: false},
	fromStringTest{"0", 0, 0, true},
	fromStringTest{"0", 10, 0, true},
	fromStringTest{"0", 16, 0, true},
	fromStringTest{"10", 0, 10, true},
	fromStringTest{"10", 10, 10, true},
	fromStringTest{"10", 16, 16, true},
	fromStringTest{"-10", 16, -16, true},
	fromStringTest{in: "0x", ok: false},
	fromStringTest{"0x10", 0, 16, true},
	fromStringTest{in: "0x10", base: 16, ok: false},
	fromStringTest{"-0x10", 0, -16, true},
}


func TestSetString(t *testing.T) {
	for i, test := range fromStringTests {
		n, ok := new(Int).SetString(test.in, test.base)
		if ok != test.ok {
			t.Errorf("#%d (input '%s') ok incorrect (should be %t)", i, test.in, test.ok)
			continue
		}
		if !ok {
			continue
		}

		if n.Cmp(new(Int).New(test.out)) != 0 {
			t.Errorf("#%d (input '%s') got: %s want: %d\n", i, test.in, n, test.out)
		}
	}
}


type divSignsTest struct {
	x, y int64
	q, r int64
}


// These examples taken from the Go Language Spec, section "Arithmetic operators"
var divSignsTests = []divSignsTest{
	divSignsTest{5, 3, 1, 2},
	divSignsTest{-5, 3, -1, -2},
	divSignsTest{5, -3, -1, 2},
	divSignsTest{-5, -3, 1, -2},
	divSignsTest{1, 2, 0, 1},
}


func TestDivSigns(t *testing.T) {
	for i, test := range divSignsTests {
		x := new(Int).New(test.x)
		y := new(Int).New(test.y)
		q, r := new(Int).Div(x, y)
		expectedQ := new(Int).New(test.q)
		expectedR := new(Int).New(test.r)

		if q.Cmp(expectedQ) != 0 || r.Cmp(expectedR) != 0 {
			t.Errorf("#%d: got (%s, %s) want (%s, %s)", i, q, r, expectedQ, expectedR)
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
	err := quick.Check(checkSetBytes, nil)
	if err != nil {
		t.Error(err)
	}
}


func checkBytes(b []byte) bool {
	b2 := new(Int).SetBytes(b).Bytes()
	return bytes.Compare(b, b2) == 0
}


func TestBytes(t *testing.T) {
	err := quick.Check(checkSetBytes, nil)
	if err != nil {
		t.Error(err)
	}
}


func checkDiv(x, y []byte) bool {
	u := new(Int).SetBytes(x)
	v := new(Int).SetBytes(y)

	if len(v.abs) == 0 {
		return true
	}

	q, r := new(Int).Div(u, v)

	if r.Cmp(v) >= 0 {
		return false
	}

	uprime := new(Int).Set(q)
	uprime.Mul(uprime, v)
	uprime.Add(uprime, r)

	return uprime.Cmp(u) == 0
}


type divTest struct {
	x, y string
	q, r string
}


var divTests = []divTest{
	divTest{
		"476217953993950760840509444250624797097991362735329973741718102894495832294430498335824897858659711275234906400899559094370964723884706254265559534144986498357",
		"9353930466774385905609975137998169297361893554149986716853295022578535724979483772383667534691121982974895531435241089241440253066816724367338287092081996",
		"50911",
		"1",
	},
	divTest{
		"11510768301994997771168",
		"1328165573307167369775",
		"8",
		"885443715537658812968",
	},
}


func TestDiv(t *testing.T) {
	err := quick.Check(checkDiv, nil)
	if err != nil {
		t.Error(err)
	}

	for i, test := range divTests {
		x, _ := new(Int).SetString(test.x, 10)
		y, _ := new(Int).SetString(test.y, 10)
		expectedQ, _ := new(Int).SetString(test.q, 10)
		expectedR, _ := new(Int).SetString(test.r, 10)

		q, r := new(Int).Div(x, y)

		if q.Cmp(expectedQ) != 0 || r.Cmp(expectedR) != 0 {
			t.Errorf("#%d got (%s, %s) want (%s, %s)", i, q, r, expectedQ, expectedR)
		}
	}
}


func TestDivStepD6(t *testing.T) {
	// See Knuth, Volume 2, section 4.3.1, exercise 21. This code exercises
	// a code path which only triggers 1 in 10^{-19} cases.

	u := &Int{false, []Word{0, 0, 1 + 1<<(_W-1), _M ^ (1 << (_W - 1))}}
	v := &Int{false, []Word{5, 2 + 1<<(_W-1), 1 << (_W - 1)}}

	q, r := new(Int).Div(u, v)
	const expectedQ64 = "18446744073709551613"
	const expectedR64 = "3138550867693340382088035895064302439801311770021610913807"
	const expectedQ32 = "4294967293"
	const expectedR32 = "39614081266355540837921718287"
	if q.String() != expectedQ64 && q.String() != expectedQ32 ||
		r.String() != expectedR64 && r.String() != expectedR32 {
		t.Errorf("got (%s, %s) want (%s, %s) or (%s, %s)", q, r, expectedQ64, expectedR64, expectedQ32, expectedR32)
	}
}


type lenTest struct {
	in  string
	out int
}


var lenTests = []lenTest{
	lenTest{"0", 0},
	lenTest{"1", 1},
	lenTest{"2", 2},
	lenTest{"4", 3},
	lenTest{"0x8000", 16},
	lenTest{"0x80000000", 32},
	lenTest{"0x800000000000", 48},
	lenTest{"0x8000000000000000", 64},
	lenTest{"0x80000000000000000000", 80},
}


func TestLen(t *testing.T) {
	for i, test := range lenTests {
		n, ok := new(Int).SetString(test.in, 0)
		if !ok {
			t.Errorf("#%d test input invalid: %s", i, test.in)
			continue
		}

		if n.Len() != test.out {
			t.Errorf("#%d got %d want %d\n", i, n.Len(), test.out)
		}
	}
}


type expTest struct {
	x, y, m string
	out     string
}


var expTests = []expTest{
	expTest{"5", "0", "", "1"},
	expTest{"-5", "0", "", "-1"},
	expTest{"5", "1", "", "5"},
	expTest{"-5", "1", "", "-5"},
	expTest{"5", "2", "", "25"},
	expTest{"1", "65537", "2", "1"},
	expTest{"0x8000000000000000", "2", "", "0x40000000000000000000000000000000"},
	expTest{"0x8000000000000000", "2", "6719", "4944"},
	expTest{"0x8000000000000000", "3", "6719", "5447"},
	expTest{"0x8000000000000000", "1000", "6719", "1603"},
	expTest{"0x8000000000000000", "1000000", "6719", "3199"},
	expTest{
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
			t.Errorf("#%d error in input", i)
			continue
		}

		z := new(Int).Exp(x, y, m)
		if z.Cmp(out) != 0 {
			t.Errorf("#%d got %s want %s", i, z, out)
		}
	}
}


func checkGcd(aBytes, bBytes []byte) bool {
	a := new(Int).SetBytes(aBytes)
	b := new(Int).SetBytes(bBytes)

	x := new(Int)
	y := new(Int)
	d := new(Int)

	GcdInt(d, x, y, a, b)
	x.Mul(x, a)
	y.Mul(y, b)
	x.Add(x, y)

	return x.Cmp(d) == 0
}


type gcdTest struct {
	a, b    int64
	d, x, y int64
}


var gcdTests = []gcdTest{
	gcdTest{120, 23, 1, -9, 47},
}


func TestGcd(t *testing.T) {
	for i, test := range gcdTests {
		a := new(Int).New(test.a)
		b := new(Int).New(test.b)

		x := new(Int)
		y := new(Int)
		d := new(Int)

		expectedX := new(Int).New(test.x)
		expectedY := new(Int).New(test.y)
		expectedD := new(Int).New(test.d)

		GcdInt(d, x, y, a, b)

		if expectedX.Cmp(x) != 0 ||
			expectedY.Cmp(y) != 0 ||
			expectedD.Cmp(d) != 0 {
			t.Errorf("#%d got (%s %s %s) want (%s %s %s)", i, x, y, d, expectedX, expectedY, expectedD)
		}
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
	for i, s := range primes {
		p, _ := new(Int).SetString(s, 10)
		if !ProbablyPrime(p, 20) {
			t.Errorf("#%d prime found to be non-prime", i)
		}
	}

	for i, s := range composites {
		c, _ := new(Int).SetString(s, 10)
		if ProbablyPrime(c, 20) {
			t.Errorf("#%d composite found to be prime", i)
		}
	}
}


type rshTest struct {
	in    string
	shift int
	out   string
}


var rshTests = []rshTest{
	rshTest{"0", 0, "0"},
	rshTest{"0", 1, "0"},
	rshTest{"0", 2, "0"},
	rshTest{"1", 0, "1"},
	rshTest{"1", 1, "0"},
	rshTest{"1", 2, "0"},
	rshTest{"2", 0, "2"},
	rshTest{"2", 1, "1"},
	rshTest{"2", 2, "0"},
	rshTest{"4294967296", 0, "4294967296"},
	rshTest{"4294967296", 1, "2147483648"},
	rshTest{"4294967296", 2, "1073741824"},
	rshTest{"18446744073709551616", 0, "18446744073709551616"},
	rshTest{"18446744073709551616", 1, "9223372036854775808"},
	rshTest{"18446744073709551616", 2, "4611686018427387904"},
	rshTest{"18446744073709551616", 64, "1"},
	rshTest{"340282366920938463463374607431768211456", 64, "18446744073709551616"},
	rshTest{"340282366920938463463374607431768211456", 128, "1"},
}


func TestRsh(t *testing.T) {
	for i, test := range rshTests {
		in, _ := new(Int).SetString(test.in, 10)
		expected, _ := new(Int).SetString(test.out, 10)
		out := new(Int).Rsh(in, test.shift)

		if out.Cmp(expected) != 0 {
			t.Errorf("#%d got %s want %s", i, out, expected)
		}
	}
}
