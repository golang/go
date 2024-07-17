// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math/bits"
	"strings"
	"testing"
)

func TestMaxBase(t *testing.T) {
	if MaxBase != len(digits) {
		t.Fatalf("%d != %d", MaxBase, len(digits))
	}
}

// log2 computes the integer binary logarithm of x.
// The result is the integer n for which 2^n <= x < 2^(n+1).
// If x == 0, the result is -1.
func log2(x Word) int {
	return bits.Len(uint(x)) - 1
}

func itoa(x nat, base int) []byte {
	// special cases
	switch {
	case base < 2:
		panic("illegal base")
	case len(x) == 0:
		return []byte("0")
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
		s[i] = digits[r]
	}

	return s[i:]
}

var strTests = []struct {
	x nat    // nat value to be converted
	b int    // conversion base
	s string // expected result
}{
	{nil, 2, "0"},
	{nat{1}, 2, "1"},
	{nat{0xc5}, 2, "11000101"},
	{nat{03271}, 8, "3271"},
	{nat{10}, 10, "10"},
	{nat{1234567890}, 10, "1234567890"},
	{nat{0xdeadbeef}, 16, "deadbeef"},
	{nat{0x229be7}, 17, "1a2b3c"},
	{nat{0x309663e6}, 32, "o9cov6"},
	{nat{0x309663e6}, 62, "TakXI"},
}

func TestString(t *testing.T) {
	// test invalid base explicitly
	var panicStr string
	func() {
		defer func() {
			panicStr = recover().(string)
		}()
		natOne.utoa(1)
	}()
	if panicStr != "invalid base" {
		t.Errorf("expected panic for invalid base")
	}

	for _, a := range strTests {
		s := string(a.x.utoa(a.b))
		if s != a.s {
			t.Errorf("string%+v\n\tgot s = %s; want %s", a, s, a.s)
		}

		x, b, _, err := nat(nil).scan(strings.NewReader(a.s), a.b, false)
		if x.cmp(a.x) != 0 {
			t.Errorf("scan%+v\n\tgot z = %v; want %v", a, x, a.x)
		}
		if b != a.b {
			t.Errorf("scan%+v\n\tgot b = %d; want %d", a, b, a.b)
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
	err   error  // expected error
	next  rune   // next character (or 0, if at EOF)
}{
	// invalid: no digits
	{"", 0, false, nil, 10, 0, errNoDigits, 0},
	{"_", 0, false, nil, 10, 0, errNoDigits, 0},
	{"?", 0, false, nil, 10, 0, errNoDigits, '?'},
	{"?", 10, false, nil, 10, 0, errNoDigits, '?'},
	{"", 10, false, nil, 10, 0, errNoDigits, 0},
	{"", 36, false, nil, 36, 0, errNoDigits, 0},
	{"", 62, false, nil, 62, 0, errNoDigits, 0},
	{"0b", 0, false, nil, 2, 0, errNoDigits, 0},
	{"0o", 0, false, nil, 8, 0, errNoDigits, 0},
	{"0x", 0, false, nil, 16, 0, errNoDigits, 0},
	{"0x_", 0, false, nil, 16, 0, errNoDigits, 0},
	{"0b2", 0, false, nil, 2, 0, errNoDigits, '2'},
	{"0B2", 0, false, nil, 2, 0, errNoDigits, '2'},
	{"0o8", 0, false, nil, 8, 0, errNoDigits, '8'},
	{"0O8", 0, false, nil, 8, 0, errNoDigits, '8'},
	{"0xg", 0, false, nil, 16, 0, errNoDigits, 'g'},
	{"0Xg", 0, false, nil, 16, 0, errNoDigits, 'g'},
	{"345", 2, false, nil, 2, 0, errNoDigits, '3'},

	// invalid: incorrect use of decimal point
	{"._", 0, true, nil, 10, 0, errNoDigits, 0},
	{".0", 0, false, nil, 10, 0, errNoDigits, '.'},
	{".0", 10, false, nil, 10, 0, errNoDigits, '.'},
	{".", 0, true, nil, 10, 0, errNoDigits, 0},
	{"0x.", 0, true, nil, 16, 0, errNoDigits, 0},
	{"0x.g", 0, true, nil, 16, 0, errNoDigits, 'g'},
	{"0x.0", 0, false, nil, 16, 0, errNoDigits, '.'},

	// invalid: incorrect use of separators
	{"_0", 0, false, nil, 10, 1, errInvalSep, 0},
	{"0_", 0, false, nil, 10, 1, errInvalSep, 0},
	{"0__0", 0, false, nil, 8, 1, errInvalSep, 0},
	{"0x___0", 0, false, nil, 16, 1, errInvalSep, 0},
	{"0_x", 0, false, nil, 10, 1, errInvalSep, 'x'},
	{"0_8", 0, false, nil, 10, 1, errInvalSep, '8'},
	{"123_.", 0, true, nat{123}, 10, 0, errInvalSep, 0},
	{"._123", 0, true, nat{123}, 10, -3, errInvalSep, 0},
	{"0b__1000", 0, false, nat{0x8}, 2, 4, errInvalSep, 0},
	{"0o60___0", 0, false, nat{0600}, 8, 3, errInvalSep, 0},
	{"0466_", 0, false, nat{0466}, 8, 3, errInvalSep, 0},
	{"01234567_8", 0, false, nat{01234567}, 8, 7, errInvalSep, '8'},
	{"1_.", 0, true, nat{1}, 10, 0, errInvalSep, 0},
	{"0._1", 0, true, nat{1}, 10, -1, errInvalSep, 0},
	{"2.7_", 0, true, nat{27}, 10, -1, errInvalSep, 0},
	{"0x1.0_", 0, true, nat{0x10}, 16, -1, errInvalSep, 0},

	// valid: separators are not accepted for base != 0
	{"0_", 10, false, nil, 10, 1, nil, '_'},
	{"1__0", 10, false, nat{1}, 10, 1, nil, '_'},
	{"0__8", 10, false, nil, 10, 1, nil, '_'},
	{"xy_z_", 36, false, nat{33*36 + 34}, 36, 2, nil, '_'},

	// valid, no decimal point
	{"0", 0, false, nil, 10, 1, nil, 0},
	{"0", 36, false, nil, 36, 1, nil, 0},
	{"0", 62, false, nil, 62, 1, nil, 0},
	{"1", 0, false, nat{1}, 10, 1, nil, 0},
	{"1", 10, false, nat{1}, 10, 1, nil, 0},
	{"0 ", 0, false, nil, 10, 1, nil, ' '},
	{"00 ", 0, false, nil, 8, 1, nil, ' '}, // octal 0
	{"0b1", 0, false, nat{1}, 2, 1, nil, 0},
	{"0B11000101", 0, false, nat{0xc5}, 2, 8, nil, 0},
	{"0B110001012", 0, false, nat{0xc5}, 2, 8, nil, '2'},
	{"07", 0, false, nat{7}, 8, 1, nil, 0},
	{"08", 0, false, nil, 10, 1, nil, '8'},
	{"08", 10, false, nat{8}, 10, 2, nil, 0},
	{"018", 0, false, nat{1}, 8, 1, nil, '8'},
	{"0o7", 0, false, nat{7}, 8, 1, nil, 0},
	{"0o18", 0, false, nat{1}, 8, 1, nil, '8'},
	{"0O17", 0, false, nat{017}, 8, 2, nil, 0},
	{"03271", 0, false, nat{03271}, 8, 4, nil, 0},
	{"10ab", 0, false, nat{10}, 10, 2, nil, 'a'},
	{"1234567890", 0, false, nat{1234567890}, 10, 10, nil, 0},
	{"A", 36, false, nat{10}, 36, 1, nil, 0},
	{"A", 37, false, nat{36}, 37, 1, nil, 0},
	{"xyz", 36, false, nat{(33*36+34)*36 + 35}, 36, 3, nil, 0},
	{"XYZ?", 36, false, nat{(33*36+34)*36 + 35}, 36, 3, nil, '?'},
	{"XYZ?", 62, false, nat{(59*62+60)*62 + 61}, 62, 3, nil, '?'},
	{"0x", 16, false, nil, 16, 1, nil, 'x'},
	{"0xdeadbeef", 0, false, nat{0xdeadbeef}, 16, 8, nil, 0},
	{"0XDEADBEEF", 0, false, nat{0xdeadbeef}, 16, 8, nil, 0},

	// valid, with decimal point
	{"0.", 0, false, nil, 10, 1, nil, '.'},
	{"0.", 10, true, nil, 10, 0, nil, 0},
	{"0.1.2", 10, true, nat{1}, 10, -1, nil, '.'},
	{".000", 10, true, nil, 10, -3, nil, 0},
	{"12.3", 10, true, nat{123}, 10, -1, nil, 0},
	{"012.345", 10, true, nat{12345}, 10, -3, nil, 0},
	{"0.1", 0, true, nat{1}, 10, -1, nil, 0},
	{"0.1", 2, true, nat{1}, 2, -1, nil, 0},
	{"0.12", 2, true, nat{1}, 2, -1, nil, '2'},
	{"0b0.1", 0, true, nat{1}, 2, -1, nil, 0},
	{"0B0.12", 0, true, nat{1}, 2, -1, nil, '2'},
	{"0o0.7", 0, true, nat{7}, 8, -1, nil, 0},
	{"0O0.78", 0, true, nat{7}, 8, -1, nil, '8'},
	{"0xdead.beef", 0, true, nat{0xdeadbeef}, 16, -4, nil, 0},

	// valid, with separators
	{"1_000", 0, false, nat{1000}, 10, 4, nil, 0},
	{"0_466", 0, false, nat{0466}, 8, 3, nil, 0},
	{"0o_600", 0, false, nat{0600}, 8, 3, nil, 0},
	{"0x_f0_0d", 0, false, nat{0xf00d}, 16, 4, nil, 0},
	{"0b1000_0001", 0, false, nat{0x81}, 2, 8, nil, 0},
	{"1_000.000_1", 0, true, nat{10000001}, 10, -4, nil, 0},
	{"0x_f00d.1e", 0, true, nat{0xf00d1e}, 16, -2, nil, 0},
	{"0x_f00d.1E2", 0, true, nat{0xf00d1e2}, 16, -3, nil, 0},
	{"0x_f00d.1eg", 0, true, nat{0xf00d1e}, 16, -2, nil, 'g'},
}

func TestScanBase(t *testing.T) {
	for _, a := range natScanTests {
		r := strings.NewReader(a.s)
		x, b, count, err := nat(nil).scan(r, a.base, a.frac)
		if errors.Is(err, a.err) {
			t.Errorf("scan%+v\n\tgot error = %v; want %v", a, err, a.err)
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
	if s := string(z.utoa(10)); s != pi {
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
	if string(x.utoa(10)) != pi {
		panic("benchmark incorrect: conversion failed")
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			x.utoa(10)
		}
	})
}

func BenchmarkScan(b *testing.B) {
	const x = 10
	for _, base := range []int{2, 8, 10, 16} {
		for _, y := range []Word{10, 100, 1000, 10000, 100000} {
			if isRaceBuilder && y > 1000 {
				continue
			}
			b.Run(fmt.Sprintf("%d/Base%d", y, base), func(b *testing.B) {
				b.StopTimer()
				var z nat
				z = z.expWW(x, y)

				s := z.utoa(base)
				if t := itoa(z, base); !bytes.Equal(s, t) {
					b.Fatalf("scanning: got %s; want %s", s, t)
				}
				b.StartTimer()

				for i := 0; i < b.N; i++ {
					z.scan(bytes.NewReader(s), base, false)
				}
			})
		}
	}
}

func BenchmarkString(b *testing.B) {
	const x = 10
	for _, base := range []int{2, 8, 10, 16} {
		for _, y := range []Word{10, 100, 1000, 10000, 100000} {
			if isRaceBuilder && y > 1000 {
				continue
			}
			b.Run(fmt.Sprintf("%d/Base%d", y, base), func(b *testing.B) {
				b.StopTimer()
				var z nat
				z = z.expWW(x, y)
				z.utoa(base) // warm divisor cache
				b.StartTimer()

				for i := 0; i < b.N; i++ {
					_ = z.utoa(base)
				}
			})
		}
	}
}

func BenchmarkLeafSize(b *testing.B) {
	for n := 0; n <= 16; n++ {
		b.Run(fmt.Sprint(n), func(b *testing.B) { LeafSizeHelper(b, 10, n) })
	}
	// Try some large lengths
	for _, n := range []int{32, 64} {
		b.Run(fmt.Sprint(n), func(b *testing.B) { LeafSizeHelper(b, 10, n) })
	}
}

func LeafSizeHelper(b *testing.B, base, size int) {
	b.StopTimer()
	originalLeafSize := leafSize
	resetTable(cacheBase10.table[:])
	leafSize = size
	b.StartTimer()

	for d := 1; d <= 10000; d *= 10 {
		b.StopTimer()
		var z nat
		z = z.expWW(Word(base), Word(d)) // build target number
		_ = z.utoa(base)                 // warm divisor cache
		b.StartTimer()

		for i := 0; i < b.N; i++ {
			_ = z.utoa(base)
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
	var p Word
	for b := 2; b <= 16; b++ {
		for p = 0; p <= 512; p++ {
			if testing.Short() && p > 10 {
				break
			}
			x := nat(nil).expWW(Word(b), p)
			xs := x.utoa(b)
			xs2 := itoa(x, b)
			if !bytes.Equal(xs, xs2) {
				t.Errorf("failed at %d ** %d in base %d: %s != %s", b, p, b, xs, xs2)
			}
		}
		if b >= 3 && testing.Short() {
			break
		}
	}
}
