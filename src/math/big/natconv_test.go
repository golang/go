// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"io"
	"strings"
	"testing"
)

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
	q := nat(nil).set(x)

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
	{nat{03271}, lowercaseDigits[:8], "3271"},
	{nat{10}, lowercaseDigits[:10], "10"},
	{nat{1234567890}, uppercaseDigits[:10], "1234567890"},
	{nat{0xdeadbeef}, lowercaseDigits[:16], "deadbeef"},
	{nat{0xdeadbeef}, uppercaseDigits[:16], "DEADBEEF"},
	{nat{0x229be7}, lowercaseDigits[:17], "1a2b3c"},
	{nat{0x309663e6}, uppercaseDigits[:32], "O9COV6"},
}

func TestString(t *testing.T) {
	// test invalid character set explicitly
	var panicStr string
	func() {
		defer func() {
			panicStr = recover().(string)
		}()
		natOne.string("0")
	}()
	if panicStr != "invalid character set length" {
		t.Errorf("expected panic for invalid character set")
	}

	for _, a := range strTests {
		s := a.x.string(a.c)
		if s != a.s {
			t.Errorf("string%+v\n\tgot s = %s; want %s", a, s, a.s)
		}

		x, b, _, err := nat(nil).scan(strings.NewReader(a.s), len(a.c), false)
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
	s     string // string to be scanned
	base  int    // input base
	frac  bool   // fraction ok
	x     nat    // expected nat
	b     int    // expected base
	count int    // expected digit count
	ok    bool   // expected success
	next  rune   // next character (or 0, if at EOF)
}{
	// error: illegal base
	{base: -1},
	{base: 37},

	// error: no mantissa
	{},
	{s: "?"},
	{base: 10},
	{base: 36},
	{s: "?", base: 10},
	{s: "0x"},
	{s: "345", base: 2},

	// error: incorrect use of decimal point
	{s: ".0"},
	{s: ".0", base: 10},
	{s: ".", base: 1},
	{s: "0x.0"},

	// no errors
	{"0", 0, false, nil, 10, 1, true, 0},
	{"0", 10, false, nil, 10, 1, true, 0},
	{"0", 36, false, nil, 36, 1, true, 0},
	{"1", 0, false, nat{1}, 10, 1, true, 0},
	{"1", 10, false, nat{1}, 10, 1, true, 0},
	{"0 ", 0, false, nil, 10, 1, true, ' '},
	{"08", 0, false, nil, 10, 1, true, '8'},
	{"08", 10, false, nat{8}, 10, 2, true, 0},
	{"018", 0, false, nat{1}, 8, 1, true, '8'},
	{"0b1", 0, false, nat{1}, 2, 1, true, 0},
	{"0b11000101", 0, false, nat{0xc5}, 2, 8, true, 0},
	{"03271", 0, false, nat{03271}, 8, 4, true, 0},
	{"10ab", 0, false, nat{10}, 10, 2, true, 'a'},
	{"1234567890", 0, false, nat{1234567890}, 10, 10, true, 0},
	{"xyz", 36, false, nat{(33*36+34)*36 + 35}, 36, 3, true, 0},
	{"xyz?", 36, false, nat{(33*36+34)*36 + 35}, 36, 3, true, '?'},
	{"0x", 16, false, nil, 16, 1, true, 'x'},
	{"0xdeadbeef", 0, false, nat{0xdeadbeef}, 16, 8, true, 0},
	{"0XDEADBEEF", 0, false, nat{0xdeadbeef}, 16, 8, true, 0},

	// no errors, decimal point
	{"0.", 0, false, nil, 10, 1, true, '.'},
	{"0.", 10, true, nil, 10, 0, true, 0},
	{"0.1.2", 10, true, nat{1}, 10, -1, true, '.'},
	{".000", 10, true, nil, 10, -3, true, 0},
	{"12.3", 10, true, nat{123}, 10, -1, true, 0},
	{"012.345", 10, true, nat{12345}, 10, -3, true, 0},
}

func TestScanBase(t *testing.T) {
	for _, a := range natScanTests {
		r := strings.NewReader(a.s)
		x, b, count, err := nat(nil).scan(r, a.base, a.frac)
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
		if count != a.count {
			t.Errorf("scan%+v\n\tgot count = %d; want %d", a, count, a.count)
		}
		next, _, err := r.ReadRune()
		if err == io.EOF {
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
	z, _, _, err := x.scan(strings.NewReader(pi), 10, false)
	if err != nil {
		t.Errorf("scanning pi: %s", err)
	}
	if s := z.decimalString(); s != pi {
		t.Errorf("scanning pi: got %s", s)
	}
}

func TestScanPiParallel(t *testing.T) {
	const n = 2
	c := make(chan int)
	for i := 0; i < n; i++ {
		go func() {
			TestScanPi(t)
			c <- 0
		}()
	}
	for i := 0; i < n; i++ {
		<-c
	}
}

func BenchmarkScanPi(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x nat
		x.scan(strings.NewReader(pi), 10, false)
	}
}

func BenchmarkStringPiParallel(b *testing.B) {
	var x nat
	x, _, _, _ = x.scan(strings.NewReader(pi), 0, false)
	if x.decimalString() != pi {
		panic("benchmark incorrect: conversion failed")
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			x.decimalString()
		}
	})
}

func BenchmarkScan10Base2(b *testing.B)     { ScanHelper(b, 2, 10, 10) }
func BenchmarkScan100Base2(b *testing.B)    { ScanHelper(b, 2, 10, 100) }
func BenchmarkScan1000Base2(b *testing.B)   { ScanHelper(b, 2, 10, 1000) }
func BenchmarkScan10000Base2(b *testing.B)  { ScanHelper(b, 2, 10, 10000) }
func BenchmarkScan100000Base2(b *testing.B) { ScanHelper(b, 2, 10, 100000) }

func BenchmarkScan10Base8(b *testing.B)     { ScanHelper(b, 8, 10, 10) }
func BenchmarkScan100Base8(b *testing.B)    { ScanHelper(b, 8, 10, 100) }
func BenchmarkScan1000Base8(b *testing.B)   { ScanHelper(b, 8, 10, 1000) }
func BenchmarkScan10000Base8(b *testing.B)  { ScanHelper(b, 8, 10, 10000) }
func BenchmarkScan100000Base8(b *testing.B) { ScanHelper(b, 8, 10, 100000) }

func BenchmarkScan10Base10(b *testing.B)     { ScanHelper(b, 10, 10, 10) }
func BenchmarkScan100Base10(b *testing.B)    { ScanHelper(b, 10, 10, 100) }
func BenchmarkScan1000Base10(b *testing.B)   { ScanHelper(b, 10, 10, 1000) }
func BenchmarkScan10000Base10(b *testing.B)  { ScanHelper(b, 10, 10, 10000) }
func BenchmarkScan100000Base10(b *testing.B) { ScanHelper(b, 10, 10, 100000) }

func BenchmarkScan10Base16(b *testing.B)     { ScanHelper(b, 16, 10, 10) }
func BenchmarkScan100Base16(b *testing.B)    { ScanHelper(b, 16, 10, 100) }
func BenchmarkScan1000Base16(b *testing.B)   { ScanHelper(b, 16, 10, 1000) }
func BenchmarkScan10000Base16(b *testing.B)  { ScanHelper(b, 16, 10, 10000) }
func BenchmarkScan100000Base16(b *testing.B) { ScanHelper(b, 16, 10, 100000) }

func ScanHelper(b *testing.B, base int, x, y Word) {
	b.StopTimer()
	var z nat
	z = z.expWW(x, y)

	var s string
	s = z.string(lowercaseDigits[:base])
	if t := toString(z, lowercaseDigits[:base]); t != s {
		b.Fatalf("scanning: got %s; want %s", s, t)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		z.scan(strings.NewReader(s), base, false)
	}
}

func BenchmarkString10Base2(b *testing.B)     { StringHelper(b, 2, 10, 10) }
func BenchmarkString100Base2(b *testing.B)    { StringHelper(b, 2, 10, 100) }
func BenchmarkString1000Base2(b *testing.B)   { StringHelper(b, 2, 10, 1000) }
func BenchmarkString10000Base2(b *testing.B)  { StringHelper(b, 2, 10, 10000) }
func BenchmarkString100000Base2(b *testing.B) { StringHelper(b, 2, 10, 100000) }

func BenchmarkString10Base8(b *testing.B)     { StringHelper(b, 8, 10, 10) }
func BenchmarkString100Base8(b *testing.B)    { StringHelper(b, 8, 10, 100) }
func BenchmarkString1000Base8(b *testing.B)   { StringHelper(b, 8, 10, 1000) }
func BenchmarkString10000Base8(b *testing.B)  { StringHelper(b, 8, 10, 10000) }
func BenchmarkString100000Base8(b *testing.B) { StringHelper(b, 8, 10, 100000) }

func BenchmarkString10Base10(b *testing.B)     { StringHelper(b, 10, 10, 10) }
func BenchmarkString100Base10(b *testing.B)    { StringHelper(b, 10, 10, 100) }
func BenchmarkString1000Base10(b *testing.B)   { StringHelper(b, 10, 10, 1000) }
func BenchmarkString10000Base10(b *testing.B)  { StringHelper(b, 10, 10, 10000) }
func BenchmarkString100000Base10(b *testing.B) { StringHelper(b, 10, 10, 100000) }

func BenchmarkString10Base16(b *testing.B)     { StringHelper(b, 16, 10, 10) }
func BenchmarkString100Base16(b *testing.B)    { StringHelper(b, 16, 10, 100) }
func BenchmarkString1000Base16(b *testing.B)   { StringHelper(b, 16, 10, 1000) }
func BenchmarkString10000Base16(b *testing.B)  { StringHelper(b, 16, 10, 10000) }
func BenchmarkString100000Base16(b *testing.B) { StringHelper(b, 16, 10, 100000) }

func StringHelper(b *testing.B, base int, x, y Word) {
	b.StopTimer()
	var z nat
	z = z.expWW(x, y)
	z.string(lowercaseDigits[:base]) // warm divisor cache
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		_ = z.string(lowercaseDigits[:base])
	}
}

func BenchmarkLeafSize0(b *testing.B)  { LeafSizeHelper(b, 10, 0) } // test without splitting
func BenchmarkLeafSize1(b *testing.B)  { LeafSizeHelper(b, 10, 1) }
func BenchmarkLeafSize2(b *testing.B)  { LeafSizeHelper(b, 10, 2) }
func BenchmarkLeafSize3(b *testing.B)  { LeafSizeHelper(b, 10, 3) }
func BenchmarkLeafSize4(b *testing.B)  { LeafSizeHelper(b, 10, 4) }
func BenchmarkLeafSize5(b *testing.B)  { LeafSizeHelper(b, 10, 5) }
func BenchmarkLeafSize6(b *testing.B)  { LeafSizeHelper(b, 10, 6) }
func BenchmarkLeafSize7(b *testing.B)  { LeafSizeHelper(b, 10, 7) }
func BenchmarkLeafSize8(b *testing.B)  { LeafSizeHelper(b, 10, 8) }
func BenchmarkLeafSize9(b *testing.B)  { LeafSizeHelper(b, 10, 9) }
func BenchmarkLeafSize10(b *testing.B) { LeafSizeHelper(b, 10, 10) }
func BenchmarkLeafSize11(b *testing.B) { LeafSizeHelper(b, 10, 11) }
func BenchmarkLeafSize12(b *testing.B) { LeafSizeHelper(b, 10, 12) }
func BenchmarkLeafSize13(b *testing.B) { LeafSizeHelper(b, 10, 13) }
func BenchmarkLeafSize14(b *testing.B) { LeafSizeHelper(b, 10, 14) }
func BenchmarkLeafSize15(b *testing.B) { LeafSizeHelper(b, 10, 15) }
func BenchmarkLeafSize16(b *testing.B) { LeafSizeHelper(b, 10, 16) }
func BenchmarkLeafSize32(b *testing.B) { LeafSizeHelper(b, 10, 32) } // try some large lengths
func BenchmarkLeafSize64(b *testing.B) { LeafSizeHelper(b, 10, 64) }

func LeafSizeHelper(b *testing.B, base Word, size int) {
	b.StopTimer()
	originalLeafSize := leafSize
	resetTable(cacheBase10.table[:])
	leafSize = size
	b.StartTimer()

	for d := 1; d <= 10000; d *= 10 {
		b.StopTimer()
		var z nat
		z = z.expWW(base, Word(d))           // build target number
		_ = z.string(lowercaseDigits[:base]) // warm divisor cache
		b.StartTimer()

		for i := 0; i < b.N; i++ {
			_ = z.string(lowercaseDigits[:base])
		}
	}

	b.StopTimer()
	resetTable(cacheBase10.table[:])
	leafSize = originalLeafSize
	b.StartTimer()
}

func resetTable(table []divisor) {
	if table != nil && table[0].bbb != nil {
		for i := 0; i < len(table); i++ {
			table[i].bbb = nil
			table[i].nbits = 0
			table[i].ndigits = 0
		}
	}
}

func TestStringPowers(t *testing.T) {
	var b, p Word
	for b = 2; b <= 16; b++ {
		for p = 0; p <= 512; p++ {
			x := nat(nil).expWW(b, p)
			xs := x.string(lowercaseDigits[:b])
			xs2 := toString(x, lowercaseDigits[:b])
			if xs != xs2 {
				t.Errorf("failed at %d ** %d in base %d: %s != %s", b, p, b, xs, xs2)
			}
		}
		if b >= 3 && testing.Short() {
			break
		}
	}
}
