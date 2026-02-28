// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"math"
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

	// 1a) the denominator of an (uninitialized) zero value is not shared with the value
	s := &zero.b
	d := zero.Denom()
	if d == s {
		t.Errorf("1a) got %s (%p) == %s (%p) want different *Int values", d, d, s, s)
	}

	// 1b) the denominator of an (uninitialized) value is a new 1 each time
	d1 := zero.Denom()
	d2 := zero.Denom()
	if d1 == d2 {
		t.Errorf("1b) got %s (%p) == %s (%p) want different *Int values", d1, d1, d2, d2)
	}

	// 1c) the denominator of an initialized zero value is shared with the value
	x := new(Rat)
	x.Set(x) // initialize x (any operation that sets x explicitly will do)
	s = &x.b
	d = x.Denom()
	if d != s {
		t.Errorf("1c) got %s (%p) != %s (%p) want identical *Int values", d, d, s, s)
	}

	// 1d) a zero value remains zero independent of denominator
	x.Denom().Set(new(Int).Neg(b))
	if x.Cmp(zero) != 0 {
		t.Errorf("1d) got %s want %s", x, zero)
	}

	// 1e) a zero value may have a denominator != 0 and != 1
	x.Num().Set(a)
	qab := new(Rat).SetFrac(a, b)
	if x.Cmp(qab) != 0 {
		t.Errorf("1e) got %s want %s", x, qab)
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

func TestFloat32Distribution(t *testing.T) {
	// Generate a distribution of (sign, mantissa, exp) values
	// broader than the float32 range, and check Rat.Float32()
	// always picks the closest float32 approximation.
	var add = []int64{
		0,
		1,
		3,
		5,
		7,
		9,
		11,
	}
	var winc, einc = uint64(5), 15 // quick test (~60ms on x86-64)
	if *long {
		winc, einc = uint64(1), 1 // soak test (~1.5s on x86-64)
	}

	for _, sign := range "+-" {
		for _, a := range add {
			for wid := uint64(0); wid < 30; wid += winc {
				b := 1<<wid + a
				if sign == '-' {
					b = -b
				}
				for exp := -150; exp < 150; exp += einc {
					num, den := NewInt(b), NewInt(1)
					if exp > 0 {
						num.Lsh(num, uint(exp))
					} else {
						den.Lsh(den, uint(-exp))
					}
					r := new(Rat).SetFrac(num, den)
					f, _ := r.Float32()

					if !checkIsBestApprox32(t, f, r) {
						// Append context information.
						t.Errorf("(input was mantissa %#x, exp %d; f = %g (%b); f ~ %g; r = %v)",
							b, exp, f, f, math.Ldexp(float64(b), exp), r)
					}

					checkNonLossyRoundtrip32(t, f)
				}
			}
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
	var winc, einc = uint64(10), 500 // quick test (~12ms on x86-64)
	if *long {
		winc, einc = uint64(1), 1 // soak test (~75s on x86-64)
	}

	for _, sign := range "+-" {
		for _, a := range add {
			for wid := uint64(0); wid < 60; wid += winc {
				b := 1<<wid + a
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

					if !checkIsBestApprox64(t, f, r) {
						// Append context information.
						t.Errorf("(input was mantissa %#x, exp %d; f = %g (%b); f ~ %g; r = %v)",
							b, exp, f, f, math.Ldexp(float64(b), exp), r)
					}

					checkNonLossyRoundtrip64(t, f)
				}
			}
		}
	}
}

// TestSetFloat64NonFinite checks that SetFloat64 of a non-finite value
// returns nil.
func TestSetFloat64NonFinite(t *testing.T) {
	for _, f := range []float64{math.NaN(), math.Inf(+1), math.Inf(-1)} {
		var r Rat
		if r2 := r.SetFloat64(f); r2 != nil {
			t.Errorf("SetFloat64(%g) was %v, want nil", f, r2)
		}
	}
}

// checkNonLossyRoundtrip32 checks that a float->Rat->float roundtrip is
// non-lossy for finite f.
func checkNonLossyRoundtrip32(t *testing.T, f float32) {
	if !isFinite(float64(f)) {
		return
	}
	r := new(Rat).SetFloat64(float64(f))
	if r == nil {
		t.Errorf("Rat.SetFloat64(float64(%g) (%b)) == nil", f, f)
		return
	}
	f2, exact := r.Float32()
	if f != f2 || !exact {
		t.Errorf("Rat.SetFloat64(float64(%g)).Float32() = %g (%b), %v, want %g (%b), %v; delta = %b",
			f, f2, f2, exact, f, f, true, f2-f)
	}
}

// checkNonLossyRoundtrip64 checks that a float->Rat->float roundtrip is
// non-lossy for finite f.
func checkNonLossyRoundtrip64(t *testing.T, f float64) {
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

// checkIsBestApprox32 checks that f is the best possible float32
// approximation of r.
// Returns true on success.
func checkIsBestApprox32(t *testing.T, f float32, r *Rat) bool {
	if math.Abs(float64(f)) >= math.MaxFloat32 {
		// Cannot check +Inf, -Inf, nor the float next to them (MaxFloat32).
		// But we have tests for these special cases.
		return true
	}

	// r must be strictly between f0 and f1, the floats bracketing f.
	f0 := math.Nextafter32(f, float32(math.Inf(-1)))
	f1 := math.Nextafter32(f, float32(math.Inf(+1)))

	// For f to be correct, r must be closer to f than to f0 or f1.
	df := delta(r, float64(f))
	df0 := delta(r, float64(f0))
	df1 := delta(r, float64(f1))
	if df.Cmp(df0) > 0 {
		t.Errorf("Rat(%v).Float32() = %g (%b), but previous float32 %g (%b) is closer", r, f, f, f0, f0)
		return false
	}
	if df.Cmp(df1) > 0 {
		t.Errorf("Rat(%v).Float32() = %g (%b), but next float32 %g (%b) is closer", r, f, f, f1, f1)
		return false
	}
	if df.Cmp(df0) == 0 && !isEven32(f) {
		t.Errorf("Rat(%v).Float32() = %g (%b); halfway should have rounded to %g (%b) instead", r, f, f, f0, f0)
		return false
	}
	if df.Cmp(df1) == 0 && !isEven32(f) {
		t.Errorf("Rat(%v).Float32() = %g (%b); halfway should have rounded to %g (%b) instead", r, f, f, f1, f1)
		return false
	}
	return true
}

// checkIsBestApprox64 checks that f is the best possible float64
// approximation of r.
// Returns true on success.
func checkIsBestApprox64(t *testing.T, f float64, r *Rat) bool {
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
	if df.Cmp(df0) == 0 && !isEven64(f) {
		t.Errorf("Rat(%v).Float64() = %g (%b); halfway should have rounded to %g (%b) instead", r, f, f, f0, f0)
		return false
	}
	if df.Cmp(df1) == 0 && !isEven64(f) {
		t.Errorf("Rat(%v).Float64() = %g (%b); halfway should have rounded to %g (%b) instead", r, f, f, f1, f1)
		return false
	}
	return true
}

func isEven32(f float32) bool { return math.Float32bits(f)&1 == 0 }
func isEven64(f float64) bool { return math.Float64bits(f)&1 == 0 }

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

func TestRatSetInt64(t *testing.T) {
	var testCases = []int64{
		0,
		1,
		-1,
		12345,
		-98765,
		math.MaxInt64,
		math.MinInt64,
	}
	var r = new(Rat)
	for i, want := range testCases {
		r.SetInt64(want)
		if !r.IsInt() {
			t.Errorf("#%d: Rat.SetInt64(%d) is not an integer", i, want)
		}
		num := r.Num()
		if !num.IsInt64() {
			t.Errorf("#%d: Rat.SetInt64(%d) numerator is not an int64", i, want)
		}
		got := num.Int64()
		if got != want {
			t.Errorf("#%d: Rat.SetInt64(%d) = %d, but expected %d", i, want, got, want)
		}
	}
}

func TestRatSetUint64(t *testing.T) {
	var testCases = []uint64{
		0,
		1,
		12345,
		^uint64(0),
	}
	var r = new(Rat)
	for i, want := range testCases {
		r.SetUint64(want)
		if !r.IsInt() {
			t.Errorf("#%d: Rat.SetUint64(%d) is not an integer", i, want)
		}
		num := r.Num()
		if !num.IsUint64() {
			t.Errorf("#%d: Rat.SetUint64(%d) numerator is not a uint64", i, want)
		}
		got := num.Uint64()
		if got != want {
			t.Errorf("#%d: Rat.SetUint64(%d) = %d, but expected %d", i, want, got, want)
		}
	}
}

func BenchmarkRatCmp(b *testing.B) {
	x, y := NewRat(4, 1), NewRat(7, 2)
	for i := 0; i < b.N; i++ {
		x.Cmp(y)
	}
}

// TestIssue34919 verifies that a Rat's denominator is not modified
// when simply accessing the Rat value.
func TestIssue34919(t *testing.T) {
	for _, acc := range []struct {
		name string
		f    func(*Rat)
	}{
		{"Float32", func(x *Rat) { x.Float32() }},
		{"Float64", func(x *Rat) { x.Float64() }},
		{"Inv", func(x *Rat) { new(Rat).Inv(x) }},
		{"Sign", func(x *Rat) { x.Sign() }},
		{"IsInt", func(x *Rat) { x.IsInt() }},
		{"Num", func(x *Rat) { x.Num() }},
		// {"Denom", func(x *Rat) { x.Denom() }}, TODO(gri) should we change the API? See issue #33792.
	} {
		// A denominator of length 0 is interpreted as 1. Make sure that
		// "materialization" of the denominator doesn't lead to setting
		// the underlying array element 0 to 1.
		r := &Rat{Int{abs: nat{991}}, Int{abs: make(nat, 0, 1)}}
		acc.f(r)
		if d := r.b.abs[:1][0]; d != 0 {
			t.Errorf("%s modified denominator: got %d, want 0", acc.name, d)
		}
	}
}

func TestDenomRace(t *testing.T) {
	x := NewRat(1, 2)
	const N = 3
	c := make(chan bool, N)
	for i := 0; i < N; i++ {
		go func() {
			// Denom (also used by Float.SetRat) used to mutate x unnecessarily,
			// provoking race reports when run in the race detector.
			x.Denom()
			new(Float).SetRat(x)
			c <- true
		}()
	}
	for i := 0; i < N; i++ {
		<-c
	}
}
