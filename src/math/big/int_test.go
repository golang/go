// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"encoding/hex"
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

func TestBinomial(t *testing.T) {
	var z Int
	for _, test := range []struct {
		n, k int64
		want string
	}{
		{0, 0, "1"},
		{0, 1, "0"},
		{1, 0, "1"},
		{1, 1, "1"},
		{1, 10, "0"},
		{4, 0, "1"},
		{4, 1, "4"},
		{4, 2, "6"},
		{4, 3, "4"},
		{4, 4, "1"},
		{10, 1, "10"},
		{10, 9, "10"},
		{10, 5, "252"},
		{11, 5, "462"},
		{11, 6, "462"},
		{100, 10, "17310309456440"},
		{100, 90, "17310309456440"},
		{1000, 10, "263409560461970212832400"},
		{1000, 990, "263409560461970212832400"},
	} {
		if got := z.Binomial(test.n, test.k).String(); got != test.want {
			t.Errorf("Binomial(%d, %d) = %s; want %s", test.n, test.k, got, test.want)
		}
	}
}

func BenchmarkBinomial(b *testing.B) {
	var z Int
	for i := b.N - 1; i >= 0; i-- {
		z.Binomial(1000, 990)
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

func norm(x nat) nat {
	i := len(x)
	for i > 0 && x[i-1] == 0 {
		i--
	}
	return x[:i]
}

func TestBits(t *testing.T) {
	for _, test := range []nat{
		nil,
		{0},
		{1},
		{0, 1, 2, 3, 4},
		{4, 3, 2, 1, 0},
		{4, 3, 2, 1, 0, 0, 0, 0},
	} {
		var z Int
		z.neg = true
		got := z.SetBits(test)
		want := norm(test)
		if got.abs.cmp(want) != 0 {
			t.Errorf("SetBits(%v) = %v; want %v", test, got.abs, want)
		}

		if got.neg {
			t.Errorf("SetBits(%v): got negative result", test)
		}

		bits := nat(z.Bits())
		if bits.cmp(want) != 0 {
			t.Errorf("%v.Bits() = %v; want %v", z.abs, bits, want)
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
	// trim leading zero bytes since Bytes() won't return them
	// (was issue 12231)
	for len(b) > 0 && b[0] == 0 {
		b = b[1:]
	}
	b2 := new(Int).SetBytes(b).Bytes()
	return bytes.Equal(b, b2)
}

func TestBytes(t *testing.T) {
	if err := quick.Check(checkBytes, nil); err != nil {
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
	// y <= 0
	{"0", "0", "", "1"},
	{"1", "0", "", "1"},
	{"-10", "0", "", "1"},
	{"1234", "-1", "", "1"},

	// m == 1
	{"0", "0", "1", "0"},
	{"1", "0", "1", "0"},
	{"-10", "0", "1", "0"},
	{"1234", "-1", "1", "0"},

	// misc
	{"5", "1", "3", "2"},
	{"5", "-7", "", "1"},
	{"-5", "-7", "", "1"},
	{"5", "0", "", "1"},
	{"-5", "0", "", "1"},
	{"5", "1", "", "5"},
	{"-5", "1", "", "-5"},
	{"-5", "1", "7", "2"},
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
	// test case for issue 8822
	{
		"-0x1BCE04427D8032319A89E5C4136456671AC620883F2C4139E57F91307C485AD2D6204F4F87A58262652DB5DBBAC72B0613E51B835E7153BEC6068F5C8D696B74DBD18FEC316AEF73985CF0475663208EB46B4F17DD9DA55367B03323E5491A70997B90C059FB34809E6EE55BCFBD5F2F52233BFE62E6AA9E4E26A1D4C2439883D14F2633D55D8AA66A1ACD5595E778AC3A280517F1157989E70C1A437B849F1877B779CC3CDDEDE2DAA6594A6C66D181A00A5F777EE60596D8773998F6E988DEAE4CCA60E4DDCF9590543C89F74F603259FCAD71660D30294FBBE6490300F78A9D63FA660DC9417B8B9DDA28BEB3977B621B988E23D4D954F322C3540541BC649ABD504C50FADFD9F0987D58A2BF689313A285E773FF02899A6EF887D1D4A0D2",
		"0xB08FFB20760FFED58FADA86DFEF71AD72AA0FA763219618FE022C197E54708BB1191C66470250FCE8879487507CEE41381CA4D932F81C2B3F1AB20B539D50DCD",
		"0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73",
		"21484252197776302499639938883777710321993113097987201050501182909581359357618579566746556372589385361683610524730509041328855066514963385522570894839035884713051640171474186548713546686476761306436434146475140156284389181808675016576845833340494848283681088886584219750554408060556769486628029028720727393293111678826356480455433909233520504112074401376133077150471237549474149190242010469539006449596611576612573955754349042329130631128234637924786466585703488460540228477440853493392086251021228087076124706778899179648655221663765993962724699135217212118535057766739392069738618682722216712319320435674779146070442",
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
			// The result should be the same as for m == 0;
			// specifically, there should be no div-zero panic.
			m = &Int{abs: nat{}} // m != nil && len(m.abs) == 0
			z2 := new(Int).Exp(x, y, m)
			if z2.Cmp(z1) != 0 {
				t.Errorf("#%d: got %s want %s", i, z2, z1)
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

	// check results in presence of aliasing (issue #11284)
	a2 := new(Int).Set(a)
	b2 := new(Int).Set(b)
	a2.binaryGCD(a2, b2) // result is same as 1st argument
	if a2.Cmp(d) != 0 {
		t.Errorf("binaryGcd(%s, %s): got d = %s, want %s", a, b, a2, d)
	}

	a2 = new(Int).Set(a)
	b2 = new(Int).Set(b)
	b2.binaryGCD(a2, b2) // result is same as 2nd argument
	if b2.Cmp(d) != 0 {
		t.Errorf("binaryGcd(%s, %s): got d = %s, want %s", a, b, b2, d)
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

	if err := quick.Check(checkGcd, nil); err != nil {
		t.Error(err)
	}
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

	// https://golang.org/issue/638
	"18699199384836356663",

	"98920366548084643601728869055592650835572950932266967461790948584315647051443",
	"94560208308847015747498523884063394671606671904944666360068158221458669711639",

	// http://primes.utm.edu/lists/small/small3.html
	"449417999055441493994709297093108513015373787049558499205492347871729927573118262811508386655998299074566974373711472560655026288668094291699357843464363003144674940345912431129144354948751003607115263071543163",
	"230975859993204150666423538988557839555560243929065415434980904258310530753006723857139742334640122533598517597674807096648905501653461687601339782814316124971547968912893214002992086353183070342498989426570593",
	"5521712099665906221540423207019333379125265462121169655563495403888449493493629943498064604536961775110765377745550377067893607246020694972959780839151452457728855382113555867743022746090187341871655890805971735385789993",
	"203956878356401977405765866929034577280193993314348263094772646453283062722701277632936616063144088173312372882677123879538709400158306567338328279154499698366071906766440037074217117805690872792848149112022286332144876183376326512083574821647933992961249917319836219304274280243803104015000563790123",

	// ECC primes: http://tools.ietf.org/html/draft-ladd-safecurves-02
	"3618502788666131106986593281521497120414687020801267626233049500247285301239",                                                                                  // Curve1174: 2^251-9
	"57896044618658097711785492504343953926634992332820282019728792003956564819949",                                                                                 // Curve25519: 2^255-19
	"9850501549098619803069760025035903451269934817616361666987073351061430442874302652853566563721228910201656997576599",                                           // E-382: 2^382-105
	"42307582002575910332922579714097346549017899709713998034217522897561970639123926132812109468141778230245837569601494931472367",                                 // Curve41417: 2^414-17
	"6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151", // E-521: 2^521-1
}

var composites = []string{
	"0",
	"1",
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

	// check that ProbablyPrime panics if n <= 0
	c := NewInt(11) // a prime
	for _, n := range []int{-1, 0, 1} {
		func() {
			defer func() {
				if n <= 0 && recover() == nil {
					t.Fatalf("expected panic from ProbablyPrime(%d)", n)
				}
			}()
			if !c.ProbablyPrime(n) {
				t.Fatalf("%v should be a prime", c)
			}
		}()
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
	{"0xff", "-0x0a", "0xf6", "-0x01", "-0xf7", "0x09"},
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

// tri generates the trinomial 2**(n*2) - 2**n - 1, which is always 3 mod 4 and
// 7 mod 8, so that 2 is always a quadratic residue.
func tri(n uint) *Int {
	x := NewInt(1)
	x.Lsh(x, n)
	x2 := new(Int).Lsh(x, n)
	x2.Sub(x2, x)
	x2.Sub(x2, intOne)
	return x2
}

func BenchmarkModSqrt225_Tonelli(b *testing.B) {
	p := tri(225)
	x := NewInt(2)
	for i := 0; i < b.N; i++ {
		x.SetUint64(2)
		x.modSqrtTonelliShanks(x, p)
	}
}

func BenchmarkModSqrt224_3Mod4(b *testing.B) {
	p := tri(225)
	x := new(Int).SetUint64(2)
	for i := 0; i < b.N; i++ {
		x.SetUint64(2)
		x.modSqrt3Mod4Prime(x, p)
	}
}

func BenchmarkModSqrt5430_Tonelli(b *testing.B) {
	p := tri(5430)
	x := new(Int).SetUint64(2)
	for i := 0; i < b.N; i++ {
		x.SetUint64(2)
		x.modSqrtTonelliShanks(x, p)
	}
}

func BenchmarkModSqrt5430_3Mod4(b *testing.B) {
	p := tri(5430)
	x := new(Int).SetUint64(2)
	for i := 0; i < b.N; i++ {
		x.SetUint64(2)
		x.modSqrt3Mod4Prime(x, p)
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
	modulus string
}{
	{"1234567", "458948883992"},
	{"239487239847", "2410312426921032588552076022197566074856950548502459942654116941958108831682612228890093858261341614673227141477904012196503648957050582631942730706805009223062734745341073406696246014589361659774041027169249453200378729434170325843778659198143763193776859869524088940195577346119843545301547043747207749969763750084308926339295559968882457872412993810129130294592999947926365264059284647209730384947211681434464714438488520940127459844288859336526896320919633919"},
}

func TestModInverse(t *testing.T) {
	var element, modulus, gcd, inverse Int
	one := NewInt(1)
	for i, test := range modInverseTests {
		(&element).SetString(test.element, 10)
		(&modulus).SetString(test.modulus, 10)
		(&inverse).ModInverse(&element, &modulus)
		(&inverse).Mul(&inverse, &element)
		(&inverse).Mod(&inverse, &modulus)
		if (&inverse).Cmp(one) != 0 {
			t.Errorf("#%d: failed (e·e^(-1)=%s)", i, &inverse)
		}
	}
	// exhaustive test for small values
	for n := 2; n < 100; n++ {
		(&modulus).SetInt64(int64(n))
		for x := 1; x < n; x++ {
			(&element).SetInt64(int64(x))
			(&gcd).GCD(nil, nil, &element, &modulus)
			if (&gcd).Cmp(one) != 0 {
				continue
			}
			(&inverse).ModInverse(&element, &modulus)
			(&inverse).Mul(&inverse, &element)
			(&inverse).Mod(&inverse, &modulus)
			if (&inverse).Cmp(one) != 0 {
				t.Errorf("ModInverse(%d,%d)*%d%%%d=%d, not 1", &element, &modulus, &element, &modulus, &inverse)
			}
		}
	}
}

// testModSqrt is a helper for TestModSqrt,
// which checks that ModSqrt can compute a square-root of elt^2.
func testModSqrt(t *testing.T, elt, mod, sq, sqrt *Int) bool {
	var sqChk, sqrtChk, sqrtsq Int
	sq.Mul(elt, elt)
	sq.Mod(sq, mod)
	z := sqrt.ModSqrt(sq, mod)
	if z != sqrt {
		t.Errorf("ModSqrt returned wrong value %s", z)
	}

	// test ModSqrt arguments outside the range [0,mod)
	sqChk.Add(sq, mod)
	z = sqrtChk.ModSqrt(&sqChk, mod)
	if z != &sqrtChk || z.Cmp(sqrt) != 0 {
		t.Errorf("ModSqrt returned inconsistent value %s", z)
	}
	sqChk.Sub(sq, mod)
	z = sqrtChk.ModSqrt(&sqChk, mod)
	if z != &sqrtChk || z.Cmp(sqrt) != 0 {
		t.Errorf("ModSqrt returned inconsistent value %s", z)
	}

	// make sure we actually got a square root
	if sqrt.Cmp(elt) == 0 {
		return true // we found the "desired" square root
	}
	sqrtsq.Mul(sqrt, sqrt) // make sure we found the "other" one
	sqrtsq.Mod(&sqrtsq, mod)
	return sq.Cmp(&sqrtsq) == 0
}

func TestModSqrt(t *testing.T) {
	var elt, mod, modx4, sq, sqrt Int
	r := rand.New(rand.NewSource(9))
	for i, s := range primes[1:] { // skip 2, use only odd primes
		mod.SetString(s, 10)
		modx4.Lsh(&mod, 2)

		// test a few random elements per prime
		for x := 1; x < 5; x++ {
			elt.Rand(r, &modx4)
			elt.Sub(&elt, &mod) // test range [-mod, 3*mod)
			if !testModSqrt(t, &elt, &mod, &sq, &sqrt) {
				t.Errorf("#%d: failed (sqrt(e) = %s)", i, &sqrt)
			}
		}
	}

	// exhaustive test for small values
	for n := 3; n < 100; n++ {
		mod.SetInt64(int64(n))
		if !mod.ProbablyPrime(10) {
			continue
		}
		isSquare := make([]bool, n)

		// test all the squares
		for x := 1; x < n; x++ {
			elt.SetInt64(int64(x))
			if !testModSqrt(t, &elt, &mod, &sq, &sqrt) {
				t.Errorf("#%d: failed (sqrt(%d,%d) = %s)", x, &elt, &mod, &sqrt)
			}
			isSquare[sq.Uint64()] = true
		}

		// test all non-squares
		for x := 1; x < n; x++ {
			sq.SetInt64(int64(x))
			z := sqrt.ModSqrt(&sq, &mod)
			if !isSquare[x] && z != nil {
				t.Errorf("#%d: failed (sqrt(%d,%d) = nil)", x, &sqrt, &mod)
			}
		}
	}
}

func TestJacobi(t *testing.T) {
	testCases := []struct {
		x, y   int64
		result int
	}{
		{0, 1, 1},
		{0, -1, 1},
		{1, 1, 1},
		{1, -1, 1},
		{0, 5, 0},
		{1, 5, 1},
		{2, 5, -1},
		{-2, 5, -1},
		{2, -5, -1},
		{-2, -5, 1},
		{3, 5, -1},
		{5, 5, 0},
		{-5, 5, 0},
		{6, 5, 1},
		{6, -5, 1},
		{-6, 5, 1},
		{-6, -5, -1},
	}

	var x, y Int

	for i, test := range testCases {
		x.SetInt64(test.x)
		y.SetInt64(test.y)
		expected := test.result
		actual := Jacobi(&x, &y)
		if actual != expected {
			t.Errorf("#%d: Jacobi(%d, %d) = %d, but expected %d", i, test.x, test.y, actual, expected)
		}
	}
}

func TestJacobiPanic(t *testing.T) {
	const failureMsg = "test failure"
	defer func() {
		msg := recover()
		if msg == nil || msg == failureMsg {
			panic(msg)
		}
		t.Log(msg)
	}()
	x := NewInt(1)
	y := NewInt(2)
	// Jacobi should panic when the second argument is even.
	Jacobi(x, y)
	panic(failureMsg)
}

func TestIssue2607(t *testing.T) {
	// This code sequence used to hang.
	n := NewInt(10)
	n.Rand(rand.New(rand.NewSource(9)), n)
}
