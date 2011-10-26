// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

var cmpTests = []struct {
	x, y nat
	r    int
}{
	{nil, nil, 0},
	{nil, nat{}, 0},
	{nat{}, nil, 0},
	{nat{}, nat{}, 0},
	{nat{0}, nat{0}, 0},
	{nat{0}, nat{1}, -1},
	{nat{1}, nat{0}, 1},
	{nat{1}, nat{1}, 0},
	{nat{0, _M}, nat{1}, 1},
	{nat{1}, nat{0, _M}, -1},
	{nat{1, _M}, nat{0, _M}, 1},
	{nat{0, _M}, nat{1, _M}, -1},
	{nat{16, 571956, 8794, 68}, nat{837, 9146, 1, 754489}, -1},
	{nat{34986, 41, 105, 1957}, nat{56, 7458, 104, 1957}, 1},
}

func TestCmp(t *testing.T) {
	for i, a := range cmpTests {
		r := a.x.cmp(a.y)
		if r != a.r {
			t.Errorf("#%d got r = %v; want %v", i, r, a.r)
		}
	}
}

type funNN func(z, x, y nat) nat
type argNN struct {
	z, x, y nat
}

var sumNN = []argNN{
	{},
	{nat{1}, nil, nat{1}},
	{nat{1111111110}, nat{123456789}, nat{987654321}},
	{nat{0, 0, 0, 1}, nil, nat{0, 0, 0, 1}},
	{nat{0, 0, 0, 1111111110}, nat{0, 0, 0, 123456789}, nat{0, 0, 0, 987654321}},
	{nat{0, 0, 0, 1}, nat{0, 0, _M}, nat{0, 0, 1}},
}

var prodNN = []argNN{
	{},
	{nil, nil, nil},
	{nil, nat{991}, nil},
	{nat{991}, nat{991}, nat{1}},
	{nat{991 * 991}, nat{991}, nat{991}},
	{nat{0, 0, 991 * 991}, nat{0, 991}, nat{0, 991}},
	{nat{1 * 991, 2 * 991, 3 * 991, 4 * 991}, nat{1, 2, 3, 4}, nat{991}},
	{nat{4, 11, 20, 30, 20, 11, 4}, nat{1, 2, 3, 4}, nat{4, 3, 2, 1}},
}

func TestSet(t *testing.T) {
	for _, a := range sumNN {
		z := nat{}.set(a.z)
		if z.cmp(a.z) != 0 {
			t.Errorf("got z = %v; want %v", z, a.z)
		}
	}
}

func testFunNN(t *testing.T, msg string, f funNN, a argNN) {
	z := f(nil, a.x, a.y)
	if z.cmp(a.z) != 0 {
		t.Errorf("%s%+v\n\tgot z = %v; want %v", msg, a, z, a.z)
	}
}

func TestFunNN(t *testing.T) {
	for _, a := range sumNN {
		arg := a
		testFunNN(t, "add", nat.add, arg)

		arg = argNN{a.z, a.y, a.x}
		testFunNN(t, "add symmetric", nat.add, arg)

		arg = argNN{a.x, a.z, a.y}
		testFunNN(t, "sub", nat.sub, arg)

		arg = argNN{a.y, a.z, a.x}
		testFunNN(t, "sub symmetric", nat.sub, arg)
	}

	for _, a := range prodNN {
		arg := a
		testFunNN(t, "mul", nat.mul, arg)

		arg = argNN{a.z, a.y, a.x}
		testFunNN(t, "mul symmetric", nat.mul, arg)
	}
}

var mulRangesN = []struct {
	a, b uint64
	prod string
}{
	{0, 0, "0"},
	{1, 1, "1"},
	{1, 2, "2"},
	{1, 3, "6"},
	{10, 10, "10"},
	{0, 100, "0"},
	{0, 1e9, "0"},
	{1, 0, "1"},                    // empty range
	{100, 1, "1"},                  // empty range
	{1, 10, "3628800"},             // 10!
	{1, 20, "2432902008176640000"}, // 20!
	{1, 100,
		"933262154439441526816992388562667004907159682643816214685929" +
			"638952175999932299156089414639761565182862536979208272237582" +
			"51185210916864000000000000000000000000", // 100!
	},
}

func TestMulRangeN(t *testing.T) {
	for i, r := range mulRangesN {
		prod := nat{}.mulRange(r.a, r.b).decimalString()
		if prod != r.prod {
			t.Errorf("#%d: got %s; want %s", i, prod, r.prod)
		}
	}
}

var mulArg, mulTmp nat

func init() {
	const n = 1000
	mulArg = make(nat, n)
	for i := 0; i < n; i++ {
		mulArg[i] = _M
	}
}

func benchmarkMulLoad() {
	for j := 1; j <= 10; j++ {
		x := mulArg[0 : j*100]
		mulTmp.mul(x, x)
	}
}

func BenchmarkMul(b *testing.B) {
	for i := 0; i < b.N; i++ {
		benchmarkMulLoad()
	}
}

func toString(x nat, charset string) string {
	base := len(charset)

	// special cases
	switch {
	case base < 2:
		panic("illegal base")
	case len(x) == 0:
		return string(charset[0])
	}

	// allocate buffer for conversion
	i := x.bitLen()/log2(Word(base)) + 1 // +1: round up
	s := make([]byte, i)

	// don't destroy x
	q := nat{}.set(x)

	// convert
	for len(q) > 0 {
		i--
		var r Word
		q, r = q.divW(q, Word(base))
		s[i] = charset[r]
	}

	return string(s[i:])
}

var strTests = []struct {
	x nat    // nat value to be converted
	c string // conversion charset
	s string // expected result
}{
	{nil, "01", "0"},
	{nat{1}, "01", "1"},
	{nat{0xc5}, "01", "11000101"},
	{nat{03271}, lowercaseDigits[0:8], "3271"},
	{nat{10}, lowercaseDigits[0:10], "10"},
	{nat{1234567890}, uppercaseDigits[0:10], "1234567890"},
	{nat{0xdeadbeef}, lowercaseDigits[0:16], "deadbeef"},
	{nat{0xdeadbeef}, uppercaseDigits[0:16], "DEADBEEF"},
	{nat{0x229be7}, lowercaseDigits[0:17], "1a2b3c"},
	{nat{0x309663e6}, uppercaseDigits[0:32], "O9COV6"},
}

func TestString(t *testing.T) {
	for _, a := range strTests {
		s := a.x.string(a.c)
		if s != a.s {
			t.Errorf("string%+v\n\tgot s = %s; want %s", a, s, a.s)
		}

		x, b, err := nat{}.scan(strings.NewReader(a.s), len(a.c))
		if x.cmp(a.x) != 0 {
			t.Errorf("scan%+v\n\tgot z = %v; want %v", a, x, a.x)
		}
		if b != len(a.c) {
			t.Errorf("scan%+v\n\tgot b = %d; want %d", a, b, len(a.c))
		}
		if err != nil {
			t.Errorf("scan%+v\n\tgot error = %s", a, err)
		}
	}
}

var natScanTests = []struct {
	s    string // string to be scanned
	base int    // input base
	x    nat    // expected nat
	b    int    // expected base
	ok   bool   // expected success
	next rune   // next character (or 0, if at EOF)
}{
	// error: illegal base
	{base: -1},
	{base: 1},
	{base: 37},

	// error: no mantissa
	{},
	{s: "?"},
	{base: 10},
	{base: 36},
	{s: "?", base: 10},
	{s: "0x"},
	{s: "345", base: 2},

	// no errors
	{"0", 0, nil, 10, true, 0},
	{"0", 10, nil, 10, true, 0},
	{"0", 36, nil, 36, true, 0},
	{"1", 0, nat{1}, 10, true, 0},
	{"1", 10, nat{1}, 10, true, 0},
	{"0 ", 0, nil, 10, true, ' '},
	{"08", 0, nil, 10, true, '8'},
	{"018", 0, nat{1}, 8, true, '8'},
	{"0b1", 0, nat{1}, 2, true, 0},
	{"0b11000101", 0, nat{0xc5}, 2, true, 0},
	{"03271", 0, nat{03271}, 8, true, 0},
	{"10ab", 0, nat{10}, 10, true, 'a'},
	{"1234567890", 0, nat{1234567890}, 10, true, 0},
	{"xyz", 36, nat{(33*36+34)*36 + 35}, 36, true, 0},
	{"xyz?", 36, nat{(33*36+34)*36 + 35}, 36, true, '?'},
	{"0x", 16, nil, 16, true, 'x'},
	{"0xdeadbeef", 0, nat{0xdeadbeef}, 16, true, 0},
	{"0XDEADBEEF", 0, nat{0xdeadbeef}, 16, true, 0},
}

func TestScanBase(t *testing.T) {
	for _, a := range natScanTests {
		r := strings.NewReader(a.s)
		x, b, err := nat{}.scan(r, a.base)
		if err == nil && !a.ok {
			t.Errorf("scan%+v\n\texpected error", a)
		}
		if err != nil {
			if a.ok {
				t.Errorf("scan%+v\n\tgot error = %s", a, err)
			}
			continue
		}
		if x.cmp(a.x) != 0 {
			t.Errorf("scan%+v\n\tgot z = %v; want %v", a, x, a.x)
		}
		if b != a.b {
			t.Errorf("scan%+v\n\tgot b = %d; want %d", a, b, a.base)
		}
		next, _, err := r.ReadRune()
		if err == os.EOF {
			next = 0
			err = nil
		}
		if err == nil && next != a.next {
			t.Errorf("scan%+v\n\tgot next = %q; want %q", a, next, a.next)
		}
	}
}

var pi = "3" +
	"14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651" +
	"32823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461" +
	"28475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920" +
	"96282925409171536436789259036001133053054882046652138414695194151160943305727036575959195309218611738193261179" +
	"31051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798" +
	"60943702770539217176293176752384674818467669405132000568127145263560827785771342757789609173637178721468440901" +
	"22495343014654958537105079227968925892354201995611212902196086403441815981362977477130996051870721134999999837" +
	"29780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083" +
	"81420617177669147303598253490428755468731159562863882353787593751957781857780532171226806613001927876611195909" +
	"21642019893809525720106548586327886593615338182796823030195203530185296899577362259941389124972177528347913151" +
	"55748572424541506959508295331168617278558890750983817546374649393192550604009277016711390098488240128583616035" +
	"63707660104710181942955596198946767837449448255379774726847104047534646208046684259069491293313677028989152104" +
	"75216205696602405803815019351125338243003558764024749647326391419927260426992279678235478163600934172164121992" +
	"45863150302861829745557067498385054945885869269956909272107975093029553211653449872027559602364806654991198818" +
	"34797753566369807426542527862551818417574672890977772793800081647060016145249192173217214772350141441973568548" +
	"16136115735255213347574184946843852332390739414333454776241686251898356948556209921922218427255025425688767179" +
	"04946016534668049886272327917860857843838279679766814541009538837863609506800642251252051173929848960841284886" +
	"26945604241965285022210661186306744278622039194945047123713786960956364371917287467764657573962413890865832645" +
	"99581339047802759009946576407895126946839835259570982582262052248940772671947826848260147699090264013639443745" +
	"53050682034962524517493996514314298091906592509372216964615157098583874105978859597729754989301617539284681382" +
	"68683868942774155991855925245953959431049972524680845987273644695848653836736222626099124608051243884390451244" +
	"13654976278079771569143599770012961608944169486855584840635342207222582848864815845602850601684273945226746767" +
	"88952521385225499546667278239864565961163548862305774564980355936345681743241125150760694794510965960940252288" +
	"79710893145669136867228748940560101503308617928680920874760917824938589009714909675985261365549781893129784821" +
	"68299894872265880485756401427047755513237964145152374623436454285844479526586782105114135473573952311342716610" +
	"21359695362314429524849371871101457654035902799344037420073105785390621983874478084784896833214457138687519435" +
	"06430218453191048481005370614680674919278191197939952061419663428754440643745123718192179998391015919561814675" +
	"14269123974894090718649423196156794520809514655022523160388193014209376213785595663893778708303906979207734672" +
	"21825625996615014215030680384477345492026054146659252014974428507325186660021324340881907104863317346496514539" +
	"05796268561005508106658796998163574736384052571459102897064140110971206280439039759515677157700420337869936007" +
	"23055876317635942187312514712053292819182618612586732157919841484882916447060957527069572209175671167229109816" +
	"90915280173506712748583222871835209353965725121083579151369882091444210067510334671103141267111369908658516398" +
	"31501970165151168517143765761835155650884909989859982387345528331635507647918535893226185489632132933089857064" +
	"20467525907091548141654985946163718027098199430992448895757128289059232332609729971208443357326548938239119325" +
	"97463667305836041428138830320382490375898524374417029132765618093773444030707469211201913020330380197621101100" +
	"44929321516084244485963766983895228684783123552658213144957685726243344189303968642624341077322697802807318915" +
	"44110104468232527162010526522721116603966655730925471105578537634668206531098965269186205647693125705863566201" +
	"85581007293606598764861179104533488503461136576867532494416680396265797877185560845529654126654085306143444318" +
	"58676975145661406800700237877659134401712749470420562230538994561314071127000407854733269939081454664645880797" +
	"27082668306343285878569830523580893306575740679545716377525420211495576158140025012622859413021647155097925923" +
	"09907965473761255176567513575178296664547791745011299614890304639947132962107340437518957359614589019389713111" +
	"79042978285647503203198691514028708085990480109412147221317947647772622414254854540332157185306142288137585043" +
	"06332175182979866223717215916077166925474873898665494945011465406284336639379003976926567214638530673609657120" +
	"91807638327166416274888800786925602902284721040317211860820419000422966171196377921337575114959501566049631862" +
	"94726547364252308177036751590673502350728354056704038674351362222477158915049530984448933309634087807693259939" +
	"78054193414473774418426312986080998886874132604721569516239658645730216315981931951673538129741677294786724229" +
	"24654366800980676928238280689964004824354037014163149658979409243237896907069779422362508221688957383798623001" +
	"59377647165122893578601588161755782973523344604281512627203734314653197777416031990665541876397929334419521541" +
	"34189948544473456738316249934191318148092777710386387734317720754565453220777092120190516609628049092636019759" +
	"88281613323166636528619326686336062735676303544776280350450777235547105859548702790814356240145171806246436267" +
	"94561275318134078330336254232783944975382437205835311477119926063813346776879695970309833913077109870408591337"

// Test case for BenchmarkScanPi.
func TestScanPi(t *testing.T) {
	var x nat
	z, _, err := x.scan(strings.NewReader(pi), 10)
	if err != nil {
		t.Errorf("scanning pi: %s", err)
	}
	if s := z.decimalString(); s != pi {
		t.Errorf("scanning pi: got %s", s)
	}
}

func BenchmarkScanPi(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x nat
		x.scan(strings.NewReader(pi), 10)
	}
}

const (
	// 314**271
	// base  2: 2249 digits
	// base  8:  751 digits
	// base 10:  678 digits
	// base 16:  563 digits
	shortBase     = 314
	shortExponent = 271

	// 3141**2178
	// base  2: 31577 digits
	// base  8: 10527 digits
	// base 10:  9507 digits
	// base 16:  7895 digits
	mediumBase     = 3141
	mediumExponent = 2718

	// 3141**2178
	// base  2: 406078 digits
	// base  8: 135360 digits
	// base 10: 122243 digits
	// base 16: 101521 digits
	longBase     = 31415
	longExponent = 27182
)

func BenchmarkScanShort2(b *testing.B) {
	ScanHelper(b, 2, shortBase, shortExponent)
}

func BenchmarkScanShort8(b *testing.B) {
	ScanHelper(b, 8, shortBase, shortExponent)
}

func BenchmarkScanSort10(b *testing.B) {
	ScanHelper(b, 10, shortBase, shortExponent)
}

func BenchmarkScanShort16(b *testing.B) {
	ScanHelper(b, 16, shortBase, shortExponent)
}

func BenchmarkScanMedium2(b *testing.B) {
	ScanHelper(b, 2, mediumBase, mediumExponent)
}

func BenchmarkScanMedium8(b *testing.B) {
	ScanHelper(b, 8, mediumBase, mediumExponent)
}

func BenchmarkScanMedium10(b *testing.B) {
	ScanHelper(b, 10, mediumBase, mediumExponent)
}

func BenchmarkScanMedium16(b *testing.B) {
	ScanHelper(b, 16, mediumBase, mediumExponent)
}

func BenchmarkScanLong2(b *testing.B) {
	ScanHelper(b, 2, longBase, longExponent)
}

func BenchmarkScanLong8(b *testing.B) {
	ScanHelper(b, 8, longBase, longExponent)
}

func BenchmarkScanLong10(b *testing.B) {
	ScanHelper(b, 10, longBase, longExponent)
}

func BenchmarkScanLong16(b *testing.B) {
	ScanHelper(b, 16, longBase, longExponent)
}

func ScanHelper(b *testing.B, base int, xv, yv Word) {
	b.StopTimer()
	var x, y, z nat
	x = x.setWord(xv)
	y = y.setWord(yv)
	z = z.expNN(x, y, nil)

	var s string
	s = z.string(lowercaseDigits[0:base])
	if t := toString(z, lowercaseDigits[0:base]); t != s {
		panic(fmt.Sprintf("scanning: got %s; want %s", s, t))
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		x.scan(strings.NewReader(s), base)
	}
}

func BenchmarkStringShort2(b *testing.B) {
	StringHelper(b, 2, shortBase, shortExponent)
}

func BenchmarkStringShort8(b *testing.B) {
	StringHelper(b, 8, shortBase, shortExponent)
}

func BenchmarkStringShort10(b *testing.B) {
	StringHelper(b, 10, shortBase, shortExponent)
}

func BenchmarkStringShort16(b *testing.B) {
	StringHelper(b, 16, shortBase, shortExponent)
}

func BenchmarkStringMedium2(b *testing.B) {
	StringHelper(b, 2, mediumBase, mediumExponent)
}

func BenchmarkStringMedium8(b *testing.B) {
	StringHelper(b, 8, mediumBase, mediumExponent)
}

func BenchmarkStringMedium10(b *testing.B) {
	StringHelper(b, 10, mediumBase, mediumExponent)
}

func BenchmarkStringMedium16(b *testing.B) {
	StringHelper(b, 16, mediumBase, mediumExponent)
}

func BenchmarkStringLong2(b *testing.B) {
	StringHelper(b, 2, longBase, longExponent)
}

func BenchmarkStringLong8(b *testing.B) {
	StringHelper(b, 8, longBase, longExponent)
}

func BenchmarkStringLong10(b *testing.B) {
	StringHelper(b, 10, longBase, longExponent)
}

func BenchmarkStringLong16(b *testing.B) {
	StringHelper(b, 16, longBase, longExponent)
}

func StringHelper(b *testing.B, base int, xv, yv Word) {
	b.StopTimer()
	var x, y, z nat
	x = x.setWord(xv)
	y = y.setWord(yv)
	z = z.expNN(x, y, nil)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		z.string(lowercaseDigits[0:base])
	}
}

func TestLeadingZeros(t *testing.T) {
	var x Word = _B >> 1
	for i := 0; i <= _W; i++ {
		if int(leadingZeros(x)) != i {
			t.Errorf("failed at %x: got %d want %d", x, leadingZeros(x), i)
		}
		x >>= 1
	}
}

type shiftTest struct {
	in    nat
	shift uint
	out   nat
}

var leftShiftTests = []shiftTest{
	{nil, 0, nil},
	{nil, 1, nil},
	{natOne, 0, natOne},
	{natOne, 1, natTwo},
	{nat{1 << (_W - 1)}, 1, nat{0}},
	{nat{1 << (_W - 1), 0}, 1, nat{0, 1}},
}

func TestShiftLeft(t *testing.T) {
	for i, test := range leftShiftTests {
		var z nat
		z = z.shl(test.in, test.shift)
		for j, d := range test.out {
			if j >= len(z) || z[j] != d {
				t.Errorf("#%d: got: %v want: %v", i, z, test.out)
				break
			}
		}
	}
}

var rightShiftTests = []shiftTest{
	{nil, 0, nil},
	{nil, 1, nil},
	{natOne, 0, natOne},
	{natOne, 1, nil},
	{natTwo, 1, natOne},
	{nat{0, 1}, 1, nat{1 << (_W - 1)}},
	{nat{2, 1, 1}, 1, nat{1<<(_W-1) + 1, 1 << (_W - 1)}},
}

func TestShiftRight(t *testing.T) {
	for i, test := range rightShiftTests {
		var z nat
		z = z.shr(test.in, test.shift)
		for j, d := range test.out {
			if j >= len(z) || z[j] != d {
				t.Errorf("#%d: got: %v want: %v", i, z, test.out)
				break
			}
		}
	}
}

type modWTest struct {
	in       string
	dividend string
	out      string
}

var modWTests32 = []modWTest{
	{"23492635982634928349238759823742", "252341", "220170"},
}

var modWTests64 = []modWTest{
	{"6527895462947293856291561095690465243862946", "524326975699234", "375066989628668"},
}

func runModWTests(t *testing.T, tests []modWTest) {
	for i, test := range tests {
		in, _ := new(Int).SetString(test.in, 10)
		d, _ := new(Int).SetString(test.dividend, 10)
		out, _ := new(Int).SetString(test.out, 10)

		r := in.abs.modW(d.abs[0])
		if r != out.abs[0] {
			t.Errorf("#%d failed: got %s want %s", i, r, out)
		}
	}
}

func TestModW(t *testing.T) {
	if _W >= 32 {
		runModWTests(t, modWTests32)
	}
	if _W >= 64 {
		runModWTests(t, modWTests64)
	}
}

func TestTrailingZeroBits(t *testing.T) {
	var x Word
	x--
	for i := 0; i < _W; i++ {
		if trailingZeroBits(x) != i {
			t.Errorf("Failed at step %d: x: %x got: %d", i, x, trailingZeroBits(x))
		}
		x <<= 1
	}
}

var expNNTests = []struct {
	x, y, m string
	out     string
}{
	{"0x8000000000000000", "2", "", "0x40000000000000000000000000000000"},
	{"0x8000000000000000", "2", "6719", "4944"},
	{"0x8000000000000000", "3", "6719", "5447"},
	{"0x8000000000000000", "1000", "6719", "1603"},
	{"0x8000000000000000", "1000000", "6719", "3199"},
	{
		"2938462938472983472983659726349017249287491026512746239764525612965293865296239471239874193284792387498274256129746192347",
		"298472983472983471903246121093472394872319615612417471234712061",
		"29834729834729834729347290846729561262544958723956495615629569234729836259263598127342374289365912465901365498236492183464",
		"23537740700184054162508175125554701713153216681790245129157191391322321508055833908509185839069455749219131480588829346291",
	},
}

func TestExpNN(t *testing.T) {
	for i, test := range expNNTests {
		x, _, _ := nat{}.scan(strings.NewReader(test.x), 0)
		y, _, _ := nat{}.scan(strings.NewReader(test.y), 0)
		out, _, _ := nat{}.scan(strings.NewReader(test.out), 0)

		var m nat

		if len(test.m) > 0 {
			m, _, _ = nat{}.scan(strings.NewReader(test.m), 0)
		}

		z := nat{}.expNN(x, y, m)
		if z.cmp(out) != 0 {
			t.Errorf("#%d got %v want %v", i, z, out)
		}
	}
}
