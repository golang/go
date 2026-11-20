// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestFloatSqrt64 tests that Float.Sqrt of numbers with 53bit mantissa
// behaves like float math.Sqrt.
func TestFloatSqrt64(t *testing.T) {
	for i := 0; i < 1e5; i++ {
		if i == 1e2 && testing.Short() {
			break
		}
		r := rand.Float64()

		got := new(Float).SetPrec(53)
		got.Sqrt(NewFloat(r))
		want := NewFloat(math.Sqrt(r))
		if got.Cmp(want) != 0 {
			t.Fatalf("Sqrt(%g) =\n got %g;\nwant %g", r, got, want)
		}
	}
}

func TestFloatSqrt(t *testing.T) {
	for _, test := range []struct {
		x    string
		want string
	}{
		// Test values were generated on Wolfram Alpha using query
		//   'sqrt(N) to 350 digits'
		// 350 decimal digits give up to 1000 binary digits.
		{"0.03125", "0.17677669529663688110021109052621225982120898442211850914708496724884155980776337985629844179095519659187673077886403712811560450698134215158051518713749197892665283324093819909447499381264409775757143376369499645074628431682460775184106467733011114982619404115381053858929018135497032545349940642599871090667456829147610370507757690729404938184321879"},
		{"0.125", "0.35355339059327376220042218105242451964241796884423701829416993449768311961552675971259688358191039318375346155772807425623120901396268430316103037427498395785330566648187639818894998762528819551514286752738999290149256863364921550368212935466022229965238808230762107717858036270994065090699881285199742181334913658295220741015515381458809876368643757"},
		{"0.5", "0.70710678118654752440084436210484903928483593768847403658833986899536623923105351942519376716382078636750692311545614851246241802792536860632206074854996791570661133296375279637789997525057639103028573505477998580298513726729843100736425870932044459930477616461524215435716072541988130181399762570399484362669827316590441482031030762917619752737287514"},
		{"2.0", "1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727350138462309122970249248360558507372126441214970999358314132226659275055927557999505011527820605714701095599716059702745345968620147285174186408891986095523292304843087143214508397626036279952514079896872533965463318088296406206152583523950547457503"},
		{"3.0", "1.7320508075688772935274463415058723669428052538103806280558069794519330169088000370811461867572485756756261414154067030299699450949989524788116555120943736485280932319023055820679748201010846749232650153123432669033228866506722546689218379712270471316603678615880190499865373798593894676503475065760507566183481296061009476021871903250831458295239598"},
		{"4.0", "2.0"},

		{"1p512", "1p256"},
		{"4p1024", "2p512"},
		{"9p2048", "3p1024"},

		{"1p-1024", "1p-512"},
		{"4p-2048", "2p-1024"},
		{"9p-4096", "3p-2048"},
	} {
		for _, prec := range []uint{24, 53, 64, 65, 100, 128, 129, 200, 256, 400, 600, 800, 1000} {
			x := new(Float).SetPrec(prec)
			x.Parse(test.x, 10)

			got := new(Float).SetPrec(prec).Sqrt(x)
			want := new(Float).SetPrec(prec)
			want.Parse(test.want, 10)
			if got.Cmp(want) != 0 {
				t.Errorf("prec = %d, Sqrt(%v) =\ngot  %g;\nwant %g",
					prec, test.x, got, want)
			}

			// Square test.
			// If got holds the square root of x to precision p, then
			//   got = √x + k
			// for some k such that |k| < 2**(-p). Thus,
			//   got² = (√x + k)² = x + 2k√n + k²
			// and the error must satisfy
			//   err = |got² - x| ≈ | 2k√n | < 2**(-p+1)*√n
			// Ignoring the k² term for simplicity.

			// err = |got² - x|
			// (but do intermediate steps with 32 guard digits to
			// avoid introducing spurious rounding-related errors)
			sq := new(Float).SetPrec(prec+32).Mul(got, got)
			diff := new(Float).Sub(sq, x)
			err := diff.Abs(diff).SetPrec(prec)

			// maxErr = 2**(-p+1)*√x
			one := new(Float).SetPrec(prec).SetInt64(1)
			maxErr := new(Float).Mul(new(Float).SetMantExp(one, -int(prec)+1), got)

			if err.Cmp(maxErr) >= 0 {
				t.Errorf("prec = %d, Sqrt(%v) =\ngot err  %g;\nwant maxErr %g",
					prec, test.x, err, maxErr)
			}
		}
	}
}

func TestFloatSqrtRoundingMode(t *testing.T) {
	// √3 = 1.732...; with 2 bits of precision the candidates are 1.5 and 2.
	x := new(Float).SetPrec(64).SetInt64(3)
	for _, test := range []struct {
		mode RoundingMode
		want string
	}{
		{ToNearestEven, "1.5"},
		{ToNearestAway, "1.5"},
		{ToZero, "1.5"},
		{AwayFromZero, "2"},
		{ToNegativeInf, "1.5"},
		{ToPositiveInf, "2"},
	} {
		z := new(Float).SetPrec(2).SetMode(test.mode).Sqrt(x)
		if got := z.Text('g', -1); got != test.want {
			t.Errorf("%s: Sqrt(3) = %s; want %s", test.mode, got, test.want)
		}
		if z.Mode() != test.mode {
			t.Errorf("%s: mode changed to %s", test.mode, z.Mode())
		}

		// The rounding mode of x must not be used.
		xm := new(Float).SetPrec(64).SetMode(test.mode).SetInt64(3)
		z = new(Float).SetPrec(2).Sqrt(xm)
		if got := z.Text('g', -1); got != "1.5" {
			t.Errorf("x.Mode() = %s: Sqrt(3) = %s; want 1.5", test.mode, got)
		}
	}

	// √1.5625 = 1.25 is the midpoint of the 2-bit floats 1 and 1.5.
	x = new(Float).SetPrec(64).SetFloat64(1.5625)
	for _, test := range []struct {
		mode RoundingMode
		want string
		acc  Accuracy
	}{
		{ToNearestEven, "1", Below},
		{ToNearestAway, "1.5", Above},
		{ToZero, "1", Below},
		{AwayFromZero, "1.5", Above},
		{ToNegativeInf, "1", Below},
		{ToPositiveInf, "1.5", Above},
	} {
		z := new(Float).SetPrec(2).SetMode(test.mode).Sqrt(x)
		if got := z.Text('g', -1); got != test.want || z.Acc() != test.acc {
			t.Errorf("%s: Sqrt(1.5625) = %s (%s); want %s (%s)", test.mode, got, z.Acc(), test.want, test.acc)
		}
	}
}

var allModes = []RoundingMode{ToNearestEven, ToNearestAway, ToZero, AwayFromZero, ToNegativeInf, ToPositiveInf}

func TestFloatSqrtExact(t *testing.T) {
	// Perfect squares must give an exact root in every mode.
	for _, root := range []string{
		"3",
		"5",
		"1.5",
		"0.25",
		"0x1p-500",
		"0x1.6a09e667f3bcc908b2fb1366ea957d3ep+500",
		"1234567890123456789012345678901234567890",
	} {
		for _, prec := range []uint{24, 53, 64, 100, 200, 300} {
			want, _, err := ParseFloat(root, 0, prec, ToNearestEven)
			if err != nil {
				t.Fatal(err)
			}
			if want.Acc() != Exact {
				continue // the root does not fit in prec bits
			}
			x := new(Float).SetPrec(2*prec).Mul(want, want)
			for _, mode := range allModes {
				z := new(Float).SetPrec(prec).SetMode(mode).Sqrt(x)
				if z.Cmp(want) != 0 || z.Acc() != Exact {
					t.Errorf("prec %d, %s: Sqrt(%s**2) = %s (%s); want %s (Exact)", prec, mode, root, z.Text('g', -1), z.Acc(), root)
				}
			}
		}
	}
}

func TestFloatSqrtDirected(t *testing.T) {
	// √2 and √10 are irrational: the directed roundings must bracket them
	// by one ulp, and the nearest roundings must pick one of the two.
	for _, xs := range []string{"2", "10", "0.3", "1e99", "0x1.8p-1001"} {
		x, _, _ := ParseFloat(xs, 0, 1000, ToNearestEven)
		for _, prec := range []uint{2, 3, 24, 53, 64, 65, 100, 300} {
			lo := new(Float).SetPrec(prec).SetMode(ToNegativeInf).Sqrt(x)
			hi := new(Float).SetPrec(prec).SetMode(ToPositiveInf).Sqrt(x)
			if lo.Acc() != Below || hi.Acc() != Above {
				t.Errorf("prec %d: Sqrt(%s): accuracies %s, %s", prec, xs, lo.Acc(), hi.Acc())
			}
			lo2 := new(Float).SetPrec(2*prec).Mul(lo, lo)
			hi2 := new(Float).SetPrec(2*prec).Mul(hi, hi)
			if lo2.Cmp(x) >= 0 || hi2.Cmp(x) <= 0 {
				t.Errorf("prec %d: Sqrt(%s): %s and %s do not bracket the root", prec, xs, lo.Text('g', -1), hi.Text('g', -1))
			}
			ulp := new(Float).SetMantExp(new(Float).SetInt64(1), lo.MantExp(nil)-int(prec))
			if d := new(Float).SetPrec(2*prec).Sub(hi, lo); d.Cmp(ulp) != 0 {
				t.Errorf("prec %d: Sqrt(%s): %s and %s are %s apart; want one ulp", prec, xs, lo.Text('g', -1), hi.Text('g', -1), d.Text('g', -1))
			}
			if z := new(Float).SetPrec(prec).SetMode(ToZero).Sqrt(x); z.Cmp(lo) != 0 {
				t.Errorf("prec %d: ToZero: Sqrt(%s) = %s; want %s", prec, xs, z.Text('g', -1), lo.Text('g', -1))
			}
			if z := new(Float).SetPrec(prec).SetMode(AwayFromZero).Sqrt(x); z.Cmp(hi) != 0 {
				t.Errorf("prec %d: AwayFromZero: Sqrt(%s) = %s; want %s", prec, xs, z.Text('g', -1), hi.Text('g', -1))
			}
			for _, mode := range []RoundingMode{ToNearestEven, ToNearestAway} {
				z := new(Float).SetPrec(prec).SetMode(mode).Sqrt(x)
				if z.Cmp(lo) != 0 && z.Cmp(hi) != 0 {
					t.Errorf("prec %d: %s: Sqrt(%s) = %s; want %s or %s", prec, mode, xs, z.Text('g', -1), lo.Text('g', -1), hi.Text('g', -1))
				}
			}
		}
	}
}

func TestFloatSqrtSpecial(t *testing.T) {
	for _, test := range []struct {
		x    *Float
		want *Float
	}{
		{NewFloat(+0), NewFloat(+0)},
		{NewFloat(-0), NewFloat(-0)},
		{NewFloat(math.Inf(+1)), NewFloat(math.Inf(+1))},
	} {
		got := new(Float).Sqrt(test.x)
		if got.neg != test.want.neg || got.form != test.want.form {
			t.Errorf("Sqrt(%v) = %v (neg: %v); want %v (neg: %v)",
				test.x, got, got.neg, test.want, test.want.neg)
		}
	}

}

// Benchmarks

func BenchmarkFloatSqrt(b *testing.B) {
	for _, prec := range []uint{64, 128, 256, 1e3, 1e4, 1e5, 1e6} {
		x := NewFloat(2)
		z := new(Float).SetPrec(prec)
		b.Run(fmt.Sprintf("%v", prec), func(b *testing.B) {
			b.ReportAllocs()
			for n := 0; n < b.N; n++ {
				z.Sqrt(x)
			}
		})
	}
}
