// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"internal/testenv"
	"math"
	"math/rand"
	"strconv"
	"strings"
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
		t.Errorf("%v %s %v\n\tgot z = %v; want %v", a.x, msg, a.y, &z, a.z)
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

	// overflow situations
	{math.MaxInt64 - 0, math.MaxInt64, "9223372036854775807"},
	{math.MaxInt64 - 1, math.MaxInt64, "85070591730234615838173535747377725442"},
	{math.MaxInt64 - 2, math.MaxInt64, "784637716923335094969050127519550606919189611815754530810"},
	{math.MaxInt64 - 3, math.MaxInt64, "7237005577332262206126809393809643289012107973151163787181513908099760521240"},
}

func TestMulRangeZ(t *testing.T) {
	var tmp Int
	// test entirely positive ranges
	for i, r := range mulRangesN {
		// skip mulRangesN entries that overflow int64
		if int64(r.a) < 0 || int64(r.b) < 0 {
			continue
		}
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
	for i := 0; i < b.N; i++ {
		z.Binomial(1000, 990)
	}
}

// Examples from the Go Language Spec, section "Arithmetic operators"
var divisionSignsTests = []struct {
	x, y int64
	q, r int64 // T-division
	d, m int64 // Euclidean division
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

func BenchmarkQuoRem(b *testing.B) {
	x, _ := new(Int).SetString("153980389784927331788354528594524332344709972855165340650588877572729725338415474372475094155672066328274535240275856844648695200875763869073572078279316458648124537905600131008790701752441155668003033945258023841165089852359980273279085783159654751552359397986180318708491098942831252291841441726305535546071", 0)
	y, _ := new(Int).SetString("7746362281539803897849273317883545285945243323447099728551653406505888775727297253384154743724750941556720663282745352402758568446486952008757638690735720782793164586481245379056001310087907017524411556680030339452580238411650898523599802732790857831596547515523593979861803187084910989428312522918414417263055355460715745539358014631136245887418412633787074173796862711588221766398229333338511838891484974940633857861775630560092874987828057333663969469797013996401149696897591265769095952887917296740109742927689053276850469671231961384715398038978492733178835452859452433234470997285516534065058887757272972533841547437247509415567206632827453524027585684464869520087576386907357207827931645864812453790560013100879070175244115566800303394525802384116508985235998027327908578315965475155235939798618031870849109894283125229184144172630553554607112725169432413343763989564437170644270643461665184965150423819594083121075825", 0)
	q := new(Int)
	r := new(Int)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q.QuoRem(y, x, r)
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
	{"1234", "-1", "0", "1"},
	{"17", "-100", "1234", "865"},
	{"2", "-100", "1234", ""},

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
	{"0x8000000000000000", "-1000000", "6719", "3663"}, // 3663 = ModInverse(3199, 6719) Issue #25865

	{"0xffffffffffffffffffffffffffffffff", "0x12345678123456781234567812345678123456789", "0x01112222333344445555666677778889", "0x36168FA1DB3AAE6C8CE647E137F97A"},

	{
		"2938462938472983472983659726349017249287491026512746239764525612965293865296239471239874193284792387498274256129746192347",
		"298472983472983471903246121093472394872319615612417471234712061",
		"29834729834729834729347290846729561262544958723956495615629569234729836259263598127342374289365912465901365498236492183464",
		"23537740700184054162508175125554701713153216681790245129157191391322321508055833908509185839069455749219131480588829346291",
	},
	// test case for issue 8822
	{
		"11001289118363089646017359372117963499250546375269047542777928006103246876688756735760905680604646624353196869572752623285140408755420374049317646428185270079555372763503115646054602867593662923894140940837479507194934267532831694565516466765025434902348314525627418515646588160955862839022051353653052947073136084780742729727874803457643848197499548297570026926927502505634297079527299004267769780768565695459945235586892627059178884998772989397505061206395455591503771677500931269477503508150175717121828518985901959919560700853226255420793148986854391552859459511723547532575574664944815966793196961286234040892865",
		"0xB08FFB20760FFED58FADA86DFEF71AD72AA0FA763219618FE022C197E54708BB1191C66470250FCE8879487507CEE41381CA4D932F81C2B3F1AB20B539D50DCD",
		"0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73",
		"21484252197776302499639938883777710321993113097987201050501182909581359357618579566746556372589385361683610524730509041328855066514963385522570894839035884713051640171474186548713546686476761306436434146475140156284389181808675016576845833340494848283681088886584219750554408060556769486628029028720727393293111678826356480455433909233520504112074401376133077150471237549474149190242010469539006449596611576612573955754349042329130631128234637924786466585703488460540228477440853493392086251021228087076124706778899179648655221663765993962724699135217212118535057766739392069738618682722216712319320435674779146070442",
	},
	{
		"-0x1BCE04427D8032319A89E5C4136456671AC620883F2C4139E57F91307C485AD2D6204F4F87A58262652DB5DBBAC72B0613E51B835E7153BEC6068F5C8D696B74DBD18FEC316AEF73985CF0475663208EB46B4F17DD9DA55367B03323E5491A70997B90C059FB34809E6EE55BCFBD5F2F52233BFE62E6AA9E4E26A1D4C2439883D14F2633D55D8AA66A1ACD5595E778AC3A280517F1157989E70C1A437B849F1877B779CC3CDDEDE2DAA6594A6C66D181A00A5F777EE60596D8773998F6E988DEAE4CCA60E4DDCF9590543C89F74F603259FCAD71660D30294FBBE6490300F78A9D63FA660DC9417B8B9DDA28BEB3977B621B988E23D4D954F322C3540541BC649ABD504C50FADFD9F0987D58A2BF689313A285E773FF02899A6EF887D1D4A0D2",
		"0xB08FFB20760FFED58FADA86DFEF71AD72AA0FA763219618FE022C197E54708BB1191C66470250FCE8879487507CEE41381CA4D932F81C2B3F1AB20B539D50DCD",
		"0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73",
		"21484252197776302499639938883777710321993113097987201050501182909581359357618579566746556372589385361683610524730509041328855066514963385522570894839035884713051640171474186548713546686476761306436434146475140156284389181808675016576845833340494848283681088886584219750554408060556769486628029028720727393293111678826356480455433909233520504112074401376133077150471237549474149190242010469539006449596611576612573955754349042329130631128234637924786466585703488460540228477440853493392086251021228087076124706778899179648655221663765993962724699135217212118535057766739392069738618682722216712319320435674779146070442",
	},

	// test cases for issue 13907
	{"0xffffffff00000001", "0xffffffff00000001", "0xffffffff00000001", "0"},
	{"0xffffffffffffffff00000001", "0xffffffffffffffff00000001", "0xffffffffffffffff00000001", "0"},
	{"0xffffffffffffffffffffffff00000001", "0xffffffffffffffffffffffff00000001", "0xffffffffffffffffffffffff00000001", "0"},
	{"0xffffffffffffffffffffffffffffffff00000001", "0xffffffffffffffffffffffffffffffff00000001", "0xffffffffffffffffffffffffffffffff00000001", "0"},

	{
		"2",
		"0xB08FFB20760FFED58FADA86DFEF71AD72AA0FA763219618FE022C197E54708BB1191C66470250FCE8879487507CEE41381CA4D932F81C2B3F1AB20B539D50DCD",
		"0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73", // odd
		"0x6AADD3E3E424D5B713FCAA8D8945B1E055166132038C57BBD2D51C833F0C5EA2007A2324CE514F8E8C2F008A2F36F44005A4039CB55830986F734C93DAF0EB4BAB54A6A8C7081864F44346E9BC6F0A3EB9F2C0146A00C6A05187D0C101E1F2D038CDB70CB5E9E05A2D188AB6CBB46286624D4415E7D4DBFAD3BCC6009D915C406EED38F468B940F41E6BEDC0430DD78E6F19A7DA3A27498A4181E24D738B0072D8F6ADB8C9809A5B033A09785814FD9919F6EF9F83EEA519BEC593855C4C10CBEEC582D4AE0792158823B0275E6AEC35242740468FAF3D5C60FD1E376362B6322F78B7ED0CA1C5BBCD2B49734A56C0967A1D01A100932C837B91D592CE08ABFF",
	},
	{
		"2",
		"0xB08FFB20760FFED58FADA86DFEF71AD72AA0FA763219618FE022C197E54708BB1191C66470250FCE8879487507CEE41381CA4D932F81C2B3F1AB20B539D50DCD",
		"0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF72", // even
		"0x7858794B5897C29F4ED0B40913416AB6C48588484E6A45F2ED3E26C941D878E923575AAC434EE2750E6439A6976F9BB4D64CEDB2A53CE8D04DD48CADCDF8E46F22747C6B81C6CEA86C0D873FBF7CEF262BAAC43A522BD7F32F3CDAC52B9337C77B3DCFB3DB3EDD80476331E82F4B1DF8EFDC1220C92656DFC9197BDC1877804E28D928A2A284B8DED506CBA304435C9D0133C246C98A7D890D1DE60CBC53A024361DA83A9B8775019083D22AC6820ED7C3C68F8E801DD4EC779EE0A05C6EB682EF9840D285B838369BA7E148FA27691D524FAEAF7C6ECE2A4B99A294B9F2C241857B5B90CC8BFFCFCF18DFA7D676131D5CD3855A5A3E8EBFA0CDFADB4D198B4A",
	},
}

func TestExp(t *testing.T) {
	for i, test := range expTests {
		x, ok1 := new(Int).SetString(test.x, 0)
		y, ok2 := new(Int).SetString(test.y, 0)

		var ok3, ok4 bool
		var out, m *Int

		if len(test.out) == 0 {
			out, ok3 = nil, true
		} else {
			out, ok3 = new(Int).SetString(test.out, 0)
		}

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
		if z1 != nil && !isNormalized(z1) {
			t.Errorf("#%d: %v is not normalized", i, *z1)
		}
		if !(z1 == nil && out == nil || z1.Cmp(out) == 0) {
			t.Errorf("#%d: got %x want %x", i, z1, out)
		}

		if m == nil {
			// The result should be the same as for m == 0;
			// specifically, there should be no div-zero panic.
			m = &Int{abs: nat{}} // m != nil && len(m.abs) == 0
			z2 := new(Int).Exp(x, y, m)
			if z2.Cmp(z1) != 0 {
				t.Errorf("#%d: got %x want %x", i, z2, z1)
			}
		}
	}
}

func BenchmarkExp(b *testing.B) {
	x, _ := new(Int).SetString("11001289118363089646017359372117963499250546375269047542777928006103246876688756735760905680604646624353196869572752623285140408755420374049317646428185270079555372763503115646054602867593662923894140940837479507194934267532831694565516466765025434902348314525627418515646588160955862839022051353653052947073136084780742729727874803457643848197499548297570026926927502505634297079527299004267769780768565695459945235586892627059178884998772989397505061206395455591503771677500931269477503508150175717121828518985901959919560700853226255420793148986854391552859459511723547532575574664944815966793196961286234040892865", 0)
	y, _ := new(Int).SetString("0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF72", 0)
	n, _ := new(Int).SetString("0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73", 0)
	out := new(Int)
	for i := 0; i < b.N; i++ {
		out.Exp(x, y, n)
	}
}

func BenchmarkExpMont(b *testing.B) {
	x, _ := new(Int).SetString("297778224889315382157302278696111964193", 0)
	y, _ := new(Int).SetString("2548977943381019743024248146923164919440527843026415174732254534318292492375775985739511369575861449426580651447974311336267954477239437734832604782764979371984246675241012538135715981292390886872929238062252506842498360562303324154310849745753254532852868768268023732398278338025070694508489163836616810661033068070127919590264734220833816416141878688318329193389865030063416339367925710474801991305827284114894677717927892032165200876093838921477120036402410731159852999623461591709308405270748511350289172153076023215", 0)
	var mods = []struct {
		name string
		val  string
	}{
		{"Odd", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF"},
		{"Even1", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FE"},
		{"Even2", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FC"},
		{"Even3", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281F8"},
		{"Even4", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281F0"},
		{"Even8", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B21828100"},
		{"Even32", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B00000000"},
		{"Even64", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828282828200FF0000000000000000"},
		{"Even96", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF82828283000000000000000000000000"},
		{"Even128", "0x82828282828200FFFF28FF2B218281FF82828282828200FFFF28FF2B218281FF00000000000000000000000000000000"},
		{"Even255", "0x82828282828200FFFF28FF2B218281FF8000000000000000000000000000000000000000000000000000000000000000"},
		{"SmallEven1", "0x7E"},
		{"SmallEven2", "0x7C"},
		{"SmallEven3", "0x78"},
		{"SmallEven4", "0x70"},
	}
	for _, mod := range mods {
		n, _ := new(Int).SetString(mod.val, 0)
		out := new(Int)
		b.Run(mod.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				out.Exp(x, y, n)
			}
		})
	}
}

func BenchmarkExp2(b *testing.B) {
	x, _ := new(Int).SetString("2", 0)
	y, _ := new(Int).SetString("0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF72", 0)
	n, _ := new(Int).SetString("0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73", 0)
	out := new(Int)
	for i := 0; i < b.N; i++ {
		out.Exp(x, y, n)
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

// euclidExtGCD is a reference implementation of Euclid's
// extended GCD algorithm for testing against optimized algorithms.
// Requirements: a, b > 0
func euclidExtGCD(a, b *Int) (g, x, y *Int) {
	A := new(Int).Set(a)
	B := new(Int).Set(b)

	// A = Ua*a + Va*b
	// B = Ub*a + Vb*b
	Ua := new(Int).SetInt64(1)
	Va := new(Int)

	Ub := new(Int)
	Vb := new(Int).SetInt64(1)

	q := new(Int)
	temp := new(Int)

	r := new(Int)
	for len(B.abs) > 0 {
		q, r = q.QuoRem(A, B, r)

		A, B, r = B, r, A

		// Ua, Ub = Ub, Ua-q*Ub
		temp.Set(Ub)
		Ub.Mul(Ub, q)
		Ub.Sub(Ua, Ub)
		Ua.Set(temp)

		// Va, Vb = Vb, Va-q*Vb
		temp.Set(Vb)
		Vb.Mul(Vb, q)
		Vb.Sub(Va, Vb)
		Va.Set(temp)
	}
	return A, Ua, Va
}

func checkLehmerGcd(aBytes, bBytes []byte) bool {
	a := new(Int).SetBytes(aBytes)
	b := new(Int).SetBytes(bBytes)

	if a.Sign() <= 0 || b.Sign() <= 0 {
		return true // can only test positive arguments
	}

	d := new(Int).lehmerGCD(nil, nil, a, b)
	d0, _, _ := euclidExtGCD(a, b)

	return d.Cmp(d0) == 0
}

func checkLehmerExtGcd(aBytes, bBytes []byte) bool {
	a := new(Int).SetBytes(aBytes)
	b := new(Int).SetBytes(bBytes)
	x := new(Int)
	y := new(Int)

	if a.Sign() <= 0 || b.Sign() <= 0 {
		return true // can only test positive arguments
	}

	d := new(Int).lehmerGCD(x, y, a, b)
	d0, x0, y0 := euclidExtGCD(a, b)

	return d.Cmp(d0) == 0 && x.Cmp(x0) == 0 && y.Cmp(y0) == 0
}

var gcdTests = []struct {
	d, x, y, a, b string
}{
	// a <= 0 || b <= 0
	{"0", "0", "0", "0", "0"},
	{"7", "0", "1", "0", "7"},
	{"7", "0", "-1", "0", "-7"},
	{"11", "1", "0", "11", "0"},
	{"7", "-1", "-2", "-77", "35"},
	{"935", "-3", "8", "64515", "24310"},
	{"935", "-3", "-8", "64515", "-24310"},
	{"935", "3", "-8", "-64515", "-24310"},

	{"1", "-9", "47", "120", "23"},
	{"7", "1", "-2", "77", "35"},
	{"935", "-3", "8", "64515", "24310"},
	{"935000000000000000", "-3", "8", "64515000000000000000", "24310000000000000000"},
	{"1", "-221", "22059940471369027483332068679400581064239780177629666810348940098015901108344", "98920366548084643601728869055592650835572950932266967461790948584315647051443", "991"},
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
		t.Errorf("GCD(%s, %s, %s, %s): got d = %s, want %s", x, y, a, b, D, d)
	}
	if x != nil && X.Cmp(x) != 0 {
		t.Errorf("GCD(%s, %s, %s, %s): got x = %s, want %s", x, y, a, b, X, x)
	}
	if y != nil && Y.Cmp(y) != 0 {
		t.Errorf("GCD(%s, %s, %s, %s): got y = %s, want %s", x, y, a, b, Y, y)
	}

	// check results in presence of aliasing (issue #11284)
	a2 := new(Int).Set(a)
	b2 := new(Int).Set(b)
	a2.GCD(X, Y, a2, b2) // result is same as 1st argument
	if a2.Cmp(d) != 0 {
		t.Errorf("aliased z = a GCD(%s, %s, %s, %s): got d = %s, want %s", x, y, a, b, a2, d)
	}
	if x != nil && X.Cmp(x) != 0 {
		t.Errorf("aliased z = a GCD(%s, %s, %s, %s): got x = %s, want %s", x, y, a, b, X, x)
	}
	if y != nil && Y.Cmp(y) != 0 {
		t.Errorf("aliased z = a GCD(%s, %s, %s, %s): got y = %s, want %s", x, y, a, b, Y, y)
	}

	a2 = new(Int).Set(a)
	b2 = new(Int).Set(b)
	b2.GCD(X, Y, a2, b2) // result is same as 2nd argument
	if b2.Cmp(d) != 0 {
		t.Errorf("aliased z = b GCD(%s, %s, %s, %s): got d = %s, want %s", x, y, a, b, b2, d)
	}
	if x != nil && X.Cmp(x) != 0 {
		t.Errorf("aliased z = b GCD(%s, %s, %s, %s): got x = %s, want %s", x, y, a, b, X, x)
	}
	if y != nil && Y.Cmp(y) != 0 {
		t.Errorf("aliased z = b GCD(%s, %s, %s, %s): got y = %s, want %s", x, y, a, b, Y, y)
	}

	a2 = new(Int).Set(a)
	b2 = new(Int).Set(b)
	D = new(Int).GCD(a2, b2, a2, b2) // x = a, y = b
	if D.Cmp(d) != 0 {
		t.Errorf("aliased x = a, y = b GCD(%s, %s, %s, %s): got d = %s, want %s", x, y, a, b, D, d)
	}
	if x != nil && a2.Cmp(x) != 0 {
		t.Errorf("aliased x = a, y = b GCD(%s, %s, %s, %s): got x = %s, want %s", x, y, a, b, a2, x)
	}
	if y != nil && b2.Cmp(y) != 0 {
		t.Errorf("aliased x = a, y = b GCD(%s, %s, %s, %s): got y = %s, want %s", x, y, a, b, b2, y)
	}

	a2 = new(Int).Set(a)
	b2 = new(Int).Set(b)
	D = new(Int).GCD(b2, a2, a2, b2) // x = b, y = a
	if D.Cmp(d) != 0 {
		t.Errorf("aliased x = b, y = a GCD(%s, %s, %s, %s): got d = %s, want %s", x, y, a, b, D, d)
	}
	if x != nil && b2.Cmp(x) != 0 {
		t.Errorf("aliased x = b, y = a GCD(%s, %s, %s, %s): got x = %s, want %s", x, y, a, b, b2, x)
	}
	if y != nil && a2.Cmp(y) != 0 {
		t.Errorf("aliased x = b, y = a GCD(%s, %s, %s, %s): got y = %s, want %s", x, y, a, b, a2, y)
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

	if err := quick.Check(checkLehmerGcd, nil); err != nil {
		t.Error(err)
	}

	if err := quick.Check(checkLehmerExtGcd, nil); err != nil {
		t.Error(err)
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

// Entries must be sorted by value in ascending order.
var cmpAbsTests = []string{
	"0",
	"1",
	"2",
	"10",
	"10000000",
	"2783678367462374683678456387645876387564783686583485",
	"2783678367462374683678456387645876387564783686583486",
	"32957394867987420967976567076075976570670947609750670956097509670576075067076027578341538",
}

func TestCmpAbs(t *testing.T) {
	values := make([]*Int, len(cmpAbsTests))
	var prev *Int
	for i, s := range cmpAbsTests {
		x, ok := new(Int).SetString(s, 0)
		if !ok {
			t.Fatalf("SetString(%s, 0) failed", s)
		}
		if prev != nil && prev.Cmp(x) >= 0 {
			t.Fatal("cmpAbsTests entries not sorted in ascending order")
		}
		values[i] = x
		prev = x
	}

	for i, x := range values {
		for j, y := range values {
			// try all combinations of signs for x, y
			for k := 0; k < 4; k++ {
				var a, b Int
				a.Set(x)
				b.Set(y)
				if k&1 != 0 {
					a.Neg(&a)
				}
				if k&2 != 0 {
					b.Neg(&b)
				}

				got := a.CmpAbs(&b)
				want := 0
				switch {
				case i > j:
					want = 1
				case i < j:
					want = -1
				}
				if got != want {
					t.Errorf("absCmp |%s|, |%s|: got %d; want %d", &a, &b, got, want)
				}
			}
		}
	}
}

func TestIntCmpSelf(t *testing.T) {
	for _, s := range cmpAbsTests {
		x, ok := new(Int).SetString(s, 0)
		if !ok {
			t.Fatalf("SetString(%s, 0) failed", s)
		}
		got := x.Cmp(x)
		want := 0
		if got != want {
			t.Errorf("x = %s: x.Cmp(x): got %d; want %d", x, got, want)
		}
	}
}

var int64Tests = []string{
	// int64
	"0",
	"1",
	"-1",
	"4294967295",
	"-4294967295",
	"4294967296",
	"-4294967296",
	"9223372036854775807",
	"-9223372036854775807",
	"-9223372036854775808",

	// not int64
	"0x8000000000000000",
	"-0x8000000000000001",
	"38579843757496759476987459679745",
	"-38579843757496759476987459679745",
}

func TestInt64(t *testing.T) {
	for _, s := range int64Tests {
		var x Int
		_, ok := x.SetString(s, 0)
		if !ok {
			t.Errorf("SetString(%s, 0) failed", s)
			continue
		}

		want, err := strconv.ParseInt(s, 0, 64)
		if err != nil {
			if err.(*strconv.NumError).Err == strconv.ErrRange {
				if x.IsInt64() {
					t.Errorf("IsInt64(%s) succeeded unexpectedly", s)
				}
			} else {
				t.Errorf("ParseInt(%s) failed", s)
			}
			continue
		}

		if !x.IsInt64() {
			t.Errorf("IsInt64(%s) failed unexpectedly", s)
		}

		got := x.Int64()
		if got != want {
			t.Errorf("Int64(%s) = %d; want %d", s, got, want)
		}
	}
}

var uint64Tests = []string{
	// uint64
	"0",
	"1",
	"4294967295",
	"4294967296",
	"8589934591",
	"8589934592",
	"9223372036854775807",
	"9223372036854775808",
	"0x08000000000000000",

	// not uint64
	"0x10000000000000000",
	"-0x08000000000000000",
	"-1",
}

func TestUint64(t *testing.T) {
	for _, s := range uint64Tests {
		var x Int
		_, ok := x.SetString(s, 0)
		if !ok {
			t.Errorf("SetString(%s, 0) failed", s)
			continue
		}

		want, err := strconv.ParseUint(s, 0, 64)
		if err != nil {
			// check for sign explicitly (ErrRange doesn't cover signed input)
			if s[0] == '-' || err.(*strconv.NumError).Err == strconv.ErrRange {
				if x.IsUint64() {
					t.Errorf("IsUint64(%s) succeeded unexpectedly", s)
				}
			} else {
				t.Errorf("ParseUint(%s) failed", s)
			}
			continue
		}

		if !x.IsUint64() {
			t.Errorf("IsUint64(%s) failed unexpectedly", s)
		}

		got := x.Uint64()
		if got != want {
			t.Errorf("Uint64(%s) = %d; want %d", s, got, want)
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

var tzbTests = []struct {
	in  string
	out uint
}{
	{"0", 0},
	{"1", 0},
	{"-1", 0},
	{"4", 2},
	{"-8", 3},
	{"0x4000000000000000000", 74},
	{"-0x8000000000000000000", 75},
}

func TestTrailingZeroBits(t *testing.T) {
	for i, test := range tzbTests {
		in, _ := new(Int).SetString(test.in, 0)
		want := test.out
		got := in.TrailingZeroBits()

		if got != want {
			t.Errorf("#%d: got %v want %v", i, got, want)
		}
	}
}

func BenchmarkBitset(b *testing.B) {
	z := new(Int)
	z.SetBit(z, 512, 1)
	b.ResetTimer()
	for i := b.N - 1; i >= 0; i-- {
		z.SetBit(z, i&512, 1)
	}
}

func BenchmarkBitsetNeg(b *testing.B) {
	z := NewInt(-1)
	z.SetBit(z, 512, 0)
	b.ResetTimer()
	for i := b.N - 1; i >= 0; i-- {
		z.SetBit(z, i&512, 0)
	}
}

func BenchmarkBitsetOrig(b *testing.B) {
	z := new(Int)
	altSetBit(z, z, 512, 1)
	b.ResetTimer()
	for i := b.N - 1; i >= 0; i-- {
		altSetBit(z, z, i&512, 1)
	}
}

func BenchmarkBitsetNegOrig(b *testing.B) {
	z := NewInt(-1)
	altSetBit(z, z, 512, 0)
	b.ResetTimer()
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

func BenchmarkModSqrt225_3Mod4(b *testing.B) {
	p := tri(225)
	x := new(Int).SetUint64(2)
	for i := 0; i < b.N; i++ {
		x.SetUint64(2)
		x.modSqrt3Mod4Prime(x, p)
	}
}

func BenchmarkModSqrt231_Tonelli(b *testing.B) {
	p := tri(231)
	p.Sub(p, intOne)
	p.Sub(p, intOne) // tri(231) - 2 is a prime == 5 mod 8
	x := new(Int).SetUint64(7)
	for i := 0; i < b.N; i++ {
		x.SetUint64(7)
		x.modSqrtTonelliShanks(x, p)
	}
}

func BenchmarkModSqrt231_5Mod8(b *testing.B) {
	p := tri(231)
	p.Sub(p, intOne)
	p.Sub(p, intOne) // tri(231) - 2 is a prime == 5 mod 8
	x := new(Int).SetUint64(7)
	for i := 0; i < b.N; i++ {
		x.SetUint64(7)
		x.modSqrt5Mod8Prime(x, p)
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
	{"-10", "13"}, // issue #16984
	{"10", "-13"},
	{"-17", "-13"},
}

func TestModInverse(t *testing.T) {
	var element, modulus, gcd, inverse Int
	one := NewInt(1)
	for _, test := range modInverseTests {
		(&element).SetString(test.element, 10)
		(&modulus).SetString(test.modulus, 10)
		(&inverse).ModInverse(&element, &modulus)
		(&inverse).Mul(&inverse, &element)
		(&inverse).Mod(&inverse, &modulus)
		if (&inverse).Cmp(one) != 0 {
			t.Errorf("ModInverse(%d,%d)*%d%%%d=%d, not 1", &element, &modulus, &element, &modulus, &inverse)
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

func BenchmarkModInverse(b *testing.B) {
	p := new(Int).SetInt64(1) // Mersenne prime 2**1279 -1
	p.abs = p.abs.lsh(p.abs, 1279)
	p.Sub(p, intOne)
	x := new(Int).Sub(p, intOne)
	z := new(Int)
	for i := 0; i < b.N; i++ {
		z.ModInverse(x, p)
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

	// test x aliasing z
	z = sqrtChk.ModSqrt(sqrtChk.Set(sq), mod)
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

		if testing.Short() && i > 2 {
			break
		}
	}

	if testing.Short() {
		return
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

func TestSqrt(t *testing.T) {
	root := 0
	r := new(Int)
	for i := 0; i < 10000; i++ {
		if (root+1)*(root+1) <= i {
			root++
		}
		n := NewInt(int64(i))
		r.SetInt64(-2)
		r.Sqrt(n)
		if r.Cmp(NewInt(int64(root))) != 0 {
			t.Errorf("Sqrt(%v) = %v, want %v", n, r, root)
		}
	}

	for i := 0; i < 1000; i += 10 {
		n, _ := new(Int).SetString("1"+strings.Repeat("0", i), 10)
		r := new(Int).Sqrt(n)
		root, _ := new(Int).SetString("1"+strings.Repeat("0", i/2), 10)
		if r.Cmp(root) != 0 {
			t.Errorf("Sqrt(1e%d) = %v, want 1e%d", i, r, i/2)
		}
	}

	// Test aliasing.
	r.SetInt64(100)
	r.Sqrt(r)
	if r.Int64() != 10 {
		t.Errorf("Sqrt(100) = %v, want 10 (aliased output)", r.Int64())
	}
}

// We can't test this together with the other Exp tests above because
// it requires a different receiver setup.
func TestIssue22830(t *testing.T) {
	one := new(Int).SetInt64(1)
	base, _ := new(Int).SetString("84555555300000000000", 10)
	mod, _ := new(Int).SetString("66666670001111111111", 10)
	want, _ := new(Int).SetString("17888885298888888889", 10)

	var tests = []int64{
		0, 1, -1,
	}

	for _, n := range tests {
		m := NewInt(n)
		if got := m.Exp(base, one, mod); got.Cmp(want) != 0 {
			t.Errorf("(%v).Exp(%s, 1, %s) = %s, want %s", n, base, mod, got, want)
		}
	}
}

func BenchmarkSqrt(b *testing.B) {
	n, _ := new(Int).SetString("1"+strings.Repeat("0", 1001), 10)
	b.ResetTimer()
	t := new(Int)
	for i := 0; i < b.N; i++ {
		t.Sqrt(n)
	}
}

func benchmarkIntSqr(b *testing.B, nwords int) {
	x := new(Int)
	x.abs = rndNat(nwords)
	t := new(Int)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t.Mul(x, x)
	}
}

func BenchmarkIntSqr(b *testing.B) {
	for _, n := range sqrBenchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			benchmarkIntSqr(b, n)
		})
	}
}

func benchmarkDiv(b *testing.B, aSize, bSize int) {
	var r = rand.New(rand.NewSource(1234))
	aa := randInt(r, uint(aSize))
	bb := randInt(r, uint(bSize))
	if aa.Cmp(bb) < 0 {
		aa, bb = bb, aa
	}
	x := new(Int)
	y := new(Int)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.DivMod(aa, bb, y)
	}
}

func BenchmarkDiv(b *testing.B) {
	sizes := []int{
		10, 20, 50, 100, 200, 500, 1000,
		1e4, 1e5, 1e6, 1e7,
	}
	for _, i := range sizes {
		j := 2 * i
		b.Run(fmt.Sprintf("%d/%d", j, i), func(b *testing.B) {
			benchmarkDiv(b, j, i)
		})
	}
}

func TestFillBytes(t *testing.T) {
	checkResult := func(t *testing.T, buf []byte, want *Int) {
		t.Helper()
		got := new(Int).SetBytes(buf)
		if got.CmpAbs(want) != 0 {
			t.Errorf("got 0x%x, want 0x%x: %x", got, want, buf)
		}
	}
	panics := func(f func()) (panic bool) {
		defer func() { panic = recover() != nil }()
		f()
		return
	}

	for _, n := range []string{
		"0",
		"1000",
		"0xffffffff",
		"-0xffffffff",
		"0xffffffffffffffff",
		"0x10000000000000000",
		"0xabababababababababababababababababababababababababa",
		"0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
	} {
		t.Run(n, func(t *testing.T) {
			t.Log(n)
			x, ok := new(Int).SetString(n, 0)
			if !ok {
				panic("invalid test entry")
			}

			// Perfectly sized buffer.
			byteLen := (x.BitLen() + 7) / 8
			buf := make([]byte, byteLen)
			checkResult(t, x.FillBytes(buf), x)

			// Way larger, checking all bytes get zeroed.
			buf = make([]byte, 100)
			for i := range buf {
				buf[i] = 0xff
			}
			checkResult(t, x.FillBytes(buf), x)

			// Too small.
			if byteLen > 0 {
				buf = make([]byte, byteLen-1)
				if !panics(func() { x.FillBytes(buf) }) {
					t.Errorf("expected panic for small buffer and value %x", x)
				}
			}
		})
	}
}

func TestNewIntMinInt64(t *testing.T) {
	// Test for uint64 cast in NewInt.
	want := int64(math.MinInt64)
	if got := NewInt(want).Int64(); got != want {
		t.Fatalf("wanted %d, got %d", want, got)
	}
}

func TestNewIntAllocs(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	for _, n := range []int64{0, 7, -7, 1 << 30, -1 << 30, 1 << 50, -1 << 50} {
		x := NewInt(3)
		got := testing.AllocsPerRun(100, func() {
			// NewInt should inline, and all its allocations
			// can happen on the stack. Passing the result of NewInt
			// to Add should not cause any of those allocations to escape.
			x.Add(x, NewInt(n))
		})
		if got != 0 {
			t.Errorf("x.Add(x, NewInt(%d)), wanted 0 allocations, got %f", n, got)
		}
	}
}

func TestFloat64(t *testing.T) {
	for _, test := range []struct {
		istr string
		f    float64
		acc  Accuracy
	}{
		{"-1000000000000000000000000000000000000000000000000000000", -1000000000000000078291540404596243842305360299886116864.000000, Below},
		{"-9223372036854775809", math.MinInt64, Above},
		{"-9223372036854775808", -9223372036854775808, Exact}, // -2^63
		{"-9223372036854775807", -9223372036854775807, Below},
		{"-18014398509481985", -18014398509481984.000000, Above},
		{"-18014398509481984", -18014398509481984.000000, Exact}, // -2^54
		{"-18014398509481983", -18014398509481984.000000, Below},
		{"-9007199254740993", -9007199254740992.000000, Above},
		{"-9007199254740992", -9007199254740992.000000, Exact}, // -2^53
		{"-9007199254740991", -9007199254740991.000000, Exact},
		{"-4503599627370497", -4503599627370497.000000, Exact},
		{"-4503599627370496", -4503599627370496.000000, Exact}, // -2^52
		{"-4503599627370495", -4503599627370495.000000, Exact},
		{"-12345", -12345, Exact},
		{"-1", -1, Exact},
		{"0", 0, Exact},
		{"1", 1, Exact},
		{"12345", 12345, Exact},
		{"0x1010000000000000", 0x1010000000000000, Exact}, // >2^53 but exact nonetheless
		{"9223372036854775807", 9223372036854775808, Above},
		{"9223372036854775808", 9223372036854775808, Exact}, // +2^63
		{"1000000000000000000000000000000000000000000000000000000", 1000000000000000078291540404596243842305360299886116864.000000, Above},
	} {
		i, ok := new(Int).SetString(test.istr, 0)
		if !ok {
			t.Errorf("SetString(%s) failed", test.istr)
			continue
		}

		// Test against expectation.
		f, acc := i.Float64()
		if f != test.f || acc != test.acc {
			t.Errorf("%s: got %f (%s); want %f (%s)", test.istr, f, acc, test.f, test.acc)
		}

		// Cross-check the fast path against the big.Float implementation.
		f2, acc2 := new(Float).SetInt(i).Float64()
		if f != f2 || acc != acc2 {
			t.Errorf("%s: got %f (%s); Float.Float64 gives %f (%s)", test.istr, f, acc, f2, acc2)
		}
	}
}
