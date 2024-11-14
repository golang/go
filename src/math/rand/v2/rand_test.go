// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"errors"
	"fmt"
	"internal/testenv"
	"math"
	. "math/rand/v2"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

const (
	numTestSamples = 10000
)

var rn, kn, wn, fn = GetNormalDistributionParameters()
var re, ke, we, fe = GetExponentialDistributionParameters()

type statsResults struct {
	mean        float64
	stddev      float64
	closeEnough float64
	maxError    float64
}

func nearEqual(a, b, closeEnough, maxError float64) bool {
	absDiff := math.Abs(a - b)
	if absDiff < closeEnough { // Necessary when one value is zero and one value is close to zero.
		return true
	}
	return absDiff/max(math.Abs(a), math.Abs(b)) < maxError
}

var testSeeds = []uint64{1, 1754801282, 1698661970, 1550503961}

// checkSimilarDistribution returns success if the mean and stddev of the
// two statsResults are similar.
func (sr *statsResults) checkSimilarDistribution(expected *statsResults) error {
	if !nearEqual(sr.mean, expected.mean, expected.closeEnough, expected.maxError) {
		s := fmt.Sprintf("mean %v != %v (allowed error %v, %v)", sr.mean, expected.mean, expected.closeEnough, expected.maxError)
		fmt.Println(s)
		return errors.New(s)
	}
	if !nearEqual(sr.stddev, expected.stddev, expected.closeEnough, expected.maxError) {
		s := fmt.Sprintf("stddev %v != %v (allowed error %v, %v)", sr.stddev, expected.stddev, expected.closeEnough, expected.maxError)
		fmt.Println(s)
		return errors.New(s)
	}
	return nil
}

func getStatsResults(samples []float64) *statsResults {
	res := new(statsResults)
	var sum, squaresum float64
	for _, s := range samples {
		sum += s
		squaresum += s * s
	}
	res.mean = sum / float64(len(samples))
	res.stddev = math.Sqrt(squaresum/float64(len(samples)) - res.mean*res.mean)
	return res
}

func checkSampleDistribution(t *testing.T, samples []float64, expected *statsResults) {
	t.Helper()
	actual := getStatsResults(samples)
	err := actual.checkSimilarDistribution(expected)
	if err != nil {
		t.Error(err)
	}
}

func checkSampleSliceDistributions(t *testing.T, samples []float64, nslices int, expected *statsResults) {
	t.Helper()
	chunk := len(samples) / nslices
	for i := 0; i < nslices; i++ {
		low := i * chunk
		var high int
		if i == nslices-1 {
			high = len(samples) - 1
		} else {
			high = (i + 1) * chunk
		}
		checkSampleDistribution(t, samples[low:high], expected)
	}
}

//
// Normal distribution tests
//

func generateNormalSamples(nsamples int, mean, stddev float64, seed uint64) []float64 {
	r := New(NewPCG(seed, seed))
	samples := make([]float64, nsamples)
	for i := range samples {
		samples[i] = r.NormFloat64()*stddev + mean
	}
	return samples
}

func testNormalDistribution(t *testing.T, nsamples int, mean, stddev float64, seed uint64) {
	//fmt.Printf("testing nsamples=%v mean=%v stddev=%v seed=%v\n", nsamples, mean, stddev, seed);

	samples := generateNormalSamples(nsamples, mean, stddev, seed)
	errorScale := max(1.0, stddev) // Error scales with stddev
	expected := &statsResults{mean, stddev, 0.10 * errorScale, 0.08 * errorScale}

	// Make sure that the entire set matches the expected distribution.
	checkSampleDistribution(t, samples, expected)

	// Make sure that each half of the set matches the expected distribution.
	checkSampleSliceDistributions(t, samples, 2, expected)

	// Make sure that each 7th of the set matches the expected distribution.
	checkSampleSliceDistributions(t, samples, 7, expected)
}

// Actual tests

func TestStandardNormalValues(t *testing.T) {
	for _, seed := range testSeeds {
		testNormalDistribution(t, numTestSamples, 0, 1, seed)
	}
}

func TestNonStandardNormalValues(t *testing.T) {
	sdmax := 1000.0
	mmax := 1000.0
	if testing.Short() {
		sdmax = 5
		mmax = 5
	}
	for sd := 0.5; sd < sdmax; sd *= 2 {
		for m := 0.5; m < mmax; m *= 2 {
			for _, seed := range testSeeds {
				testNormalDistribution(t, numTestSamples, m, sd, seed)
				if testing.Short() {
					break
				}
			}
		}
	}
}

//
// Exponential distribution tests
//

func generateExponentialSamples(nsamples int, rate float64, seed uint64) []float64 {
	r := New(NewPCG(seed, seed))
	samples := make([]float64, nsamples)
	for i := range samples {
		samples[i] = r.ExpFloat64() / rate
	}
	return samples
}

func testExponentialDistribution(t *testing.T, nsamples int, rate float64, seed uint64) {
	//fmt.Printf("testing nsamples=%v rate=%v seed=%v\n", nsamples, rate, seed);

	mean := 1 / rate
	stddev := mean

	samples := generateExponentialSamples(nsamples, rate, seed)
	errorScale := max(1.0, 1/rate) // Error scales with the inverse of the rate
	expected := &statsResults{mean, stddev, 0.10 * errorScale, 0.20 * errorScale}

	// Make sure that the entire set matches the expected distribution.
	checkSampleDistribution(t, samples, expected)

	// Make sure that each half of the set matches the expected distribution.
	checkSampleSliceDistributions(t, samples, 2, expected)

	// Make sure that each 7th of the set matches the expected distribution.
	checkSampleSliceDistributions(t, samples, 7, expected)
}

// Actual tests

func TestStandardExponentialValues(t *testing.T) {
	for _, seed := range testSeeds {
		testExponentialDistribution(t, numTestSamples, 1, seed)
	}
}

func TestNonStandardExponentialValues(t *testing.T) {
	for rate := 0.05; rate < 10; rate *= 2 {
		for _, seed := range testSeeds {
			testExponentialDistribution(t, numTestSamples, rate, seed)
			if testing.Short() {
				break
			}
		}
	}
}

//
// Table generation tests
//

func initNorm() (testKn []uint32, testWn, testFn []float32) {
	const m1 = 1 << 31
	var (
		dn float64 = rn
		tn         = dn
		vn float64 = 9.91256303526217e-3
	)

	testKn = make([]uint32, 128)
	testWn = make([]float32, 128)
	testFn = make([]float32, 128)

	q := vn / math.Exp(-0.5*dn*dn)
	testKn[0] = uint32((dn / q) * m1)
	testKn[1] = 0
	testWn[0] = float32(q / m1)
	testWn[127] = float32(dn / m1)
	testFn[0] = 1.0
	testFn[127] = float32(math.Exp(-0.5 * dn * dn))
	for i := 126; i >= 1; i-- {
		dn = math.Sqrt(-2.0 * math.Log(vn/dn+math.Exp(-0.5*dn*dn)))
		testKn[i+1] = uint32((dn / tn) * m1)
		tn = dn
		testFn[i] = float32(math.Exp(-0.5 * dn * dn))
		testWn[i] = float32(dn / m1)
	}
	return
}

func initExp() (testKe []uint32, testWe, testFe []float32) {
	const m2 = 1 << 32
	var (
		de float64 = re
		te         = de
		ve float64 = 3.9496598225815571993e-3
	)

	testKe = make([]uint32, 256)
	testWe = make([]float32, 256)
	testFe = make([]float32, 256)

	q := ve / math.Exp(-de)
	testKe[0] = uint32((de / q) * m2)
	testKe[1] = 0
	testWe[0] = float32(q / m2)
	testWe[255] = float32(de / m2)
	testFe[0] = 1.0
	testFe[255] = float32(math.Exp(-de))
	for i := 254; i >= 1; i-- {
		de = -math.Log(ve/de + math.Exp(-de))
		testKe[i+1] = uint32((de / te) * m2)
		te = de
		testFe[i] = float32(math.Exp(-de))
		testWe[i] = float32(de / m2)
	}
	return
}

// compareUint32Slices returns the first index where the two slices
// disagree, or <0 if the lengths are the same and all elements
// are identical.
func compareUint32Slices(s1, s2 []uint32) int {
	if len(s1) != len(s2) {
		if len(s1) > len(s2) {
			return len(s2) + 1
		}
		return len(s1) + 1
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			return i
		}
	}
	return -1
}

// compareFloat32Slices returns the first index where the two slices
// disagree, or <0 if the lengths are the same and all elements
// are identical.
func compareFloat32Slices(s1, s2 []float32) int {
	if len(s1) != len(s2) {
		if len(s1) > len(s2) {
			return len(s2) + 1
		}
		return len(s1) + 1
	}
	for i := range s1 {
		if !nearEqual(float64(s1[i]), float64(s2[i]), 0, 1e-7) {
			return i
		}
	}
	return -1
}

func TestNormTables(t *testing.T) {
	testKn, testWn, testFn := initNorm()
	if i := compareUint32Slices(kn[0:], testKn); i >= 0 {
		t.Errorf("kn disagrees at index %v; %v != %v", i, kn[i], testKn[i])
	}
	if i := compareFloat32Slices(wn[0:], testWn); i >= 0 {
		t.Errorf("wn disagrees at index %v; %v != %v", i, wn[i], testWn[i])
	}
	if i := compareFloat32Slices(fn[0:], testFn); i >= 0 {
		t.Errorf("fn disagrees at index %v; %v != %v", i, fn[i], testFn[i])
	}
}

func TestExpTables(t *testing.T) {
	testKe, testWe, testFe := initExp()
	if i := compareUint32Slices(ke[0:], testKe); i >= 0 {
		t.Errorf("ke disagrees at index %v; %v != %v", i, ke[i], testKe[i])
	}
	if i := compareFloat32Slices(we[0:], testWe); i >= 0 {
		t.Errorf("we disagrees at index %v; %v != %v", i, we[i], testWe[i])
	}
	if i := compareFloat32Slices(fe[0:], testFe); i >= 0 {
		t.Errorf("fe disagrees at index %v; %v != %v", i, fe[i], testFe[i])
	}
}

func hasSlowFloatingPoint() bool {
	switch runtime.GOARCH {
	case "arm":
		return os.Getenv("GOARM") == "5"
	case "mips", "mipsle", "mips64", "mips64le":
		// Be conservative and assume that all mips boards
		// have emulated floating point.
		// TODO: detect what it actually has.
		return true
	}
	return false
}

func TestFloat32(t *testing.T) {
	// For issue 6721, the problem came after 7533753 calls, so check 10e6.
	num := int(10e6)
	// But do the full amount only on builders (not locally).
	// But ARM5 floating point emulation is slow (Issue 10749), so
	// do less for that builder:
	if testing.Short() && (testenv.Builder() == "" || hasSlowFloatingPoint()) {
		num /= 100 // 1.72 seconds instead of 172 seconds
	}

	r := testRand()
	for ct := 0; ct < num; ct++ {
		f := r.Float32()
		if f >= 1 {
			t.Fatal("Float32() should be in range [0,1). ct:", ct, "f:", f)
		}
	}
}

func TestShuffleSmall(t *testing.T) {
	// Check that Shuffle allows n=0 and n=1, but that swap is never called for them.
	r := testRand()
	for n := 0; n <= 1; n++ {
		r.Shuffle(n, func { i, j -> t.Fatalf("swap called, n=%d i=%d j=%d", n, i, j) })
	}
}

// encodePerm converts from a permuted slice of length n, such as Perm generates, to an int in [0, n!).
// See https://en.wikipedia.org/wiki/Lehmer_code.
// encodePerm modifies the input slice.
func encodePerm(s []int) int {
	// Convert to Lehmer code.
	for i, x := range s {
		r := s[i+1:]
		for j, y := range r {
			if y > x {
				r[j]--
			}
		}
	}
	// Convert to int in [0, n!).
	m := 0
	fact := 1
	for i := len(s) - 1; i >= 0; i-- {
		m += s[i] * fact
		fact *= len(s) - i
	}
	return m
}

// TestUniformFactorial tests several ways of generating a uniform value in [0, n!).
func TestUniformFactorial(t *testing.T) {
	r := New(NewPCG(1, 2))
	top := 6
	if testing.Short() {
		top = 3
	}
	for n := 3; n <= top; n++ {
		t.Run(fmt.Sprintf("n=%d", n), func { t ->
			// Calculate n!.
			nfact := 1
			for i := 2; i <= n; i++ {
				nfact *= i
			}

			// Test a few different ways to generate a uniform distribution.
			p := make([]int, n) // re-usable slice for Shuffle generator
			tests := [...]struct {
				name string
				fn   func() int
			}{
				{name: "Int32N", fn: func() int { return int(r.Int32N(int32(nfact))) }},
				{name: "Perm", fn: func() int { return encodePerm(r.Perm(n)) }},
				{name: "Shuffle", fn: func() int {
					// Generate permutation using Shuffle.
					for i := range p {
						p[i] = i
					}
					r.Shuffle(n, func { i, j -> p[i], p[j] = p[j], p[i] })
					return encodePerm(p)
				}},
			}

			for _, test := range tests {
				t.Run(test.name, func { t ->
					// Gather chi-squared values and check that they follow
					// the expected normal distribution given n!-1 degrees of freedom.
					// See https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test and
					// https://www.johndcook.com/Beautiful_Testing_ch10.pdf.
					nsamples := 10 * nfact
					if nsamples < 1000 {
						nsamples = 1000
					}
					samples := make([]float64, nsamples)
					for i := range samples {
						// Generate some uniformly distributed values and count their occurrences.
						const iters = 1000
						counts := make([]int, nfact)
						for i := 0; i < iters; i++ {
							counts[test.fn()]++
						}
						// Calculate chi-squared and add to samples.
						want := iters / float64(nfact)
						var χ2 float64
						for _, have := range counts {
							err := float64(have) - want
							χ2 += err * err
						}
						χ2 /= want
						samples[i] = χ2
					}

					// Check that our samples approximate the appropriate normal distribution.
					dof := float64(nfact - 1)
					expected := &statsResults{mean: dof, stddev: math.Sqrt(2 * dof)}
					errorScale := max(1.0, expected.stddev)
					expected.closeEnough = 0.10 * errorScale
					expected.maxError = 0.08 // TODO: What is the right value here? See issue 21211.
					checkSampleDistribution(t, samples, expected)
				})
			}
		})
	}
}

// Benchmarks

var Sink uint64

func testRand() *Rand {
	return New(NewPCG(1, 2))
}

func BenchmarkSourceUint64(b *testing.B) {
	s := NewPCG(1, 2)
	var t uint64
	for n := b.N; n > 0; n-- {
		t += s.Uint64()
	}
	Sink = uint64(t)
}

func BenchmarkGlobalInt64(b *testing.B) {
	var t int64
	for n := b.N; n > 0; n-- {
		t += Int64()
	}
	Sink = uint64(t)
}

func BenchmarkGlobalInt64Parallel(b *testing.B) {
	b.RunParallel(func { pb ->
		var t int64
		for pb.Next() {
			t += Int64()
		}
		atomic.AddUint64(&Sink, uint64(t))
	})
}

func BenchmarkGlobalUint64(b *testing.B) {
	var t uint64
	for n := b.N; n > 0; n-- {
		t += Uint64()
	}
	Sink = t
}

func BenchmarkGlobalUint64Parallel(b *testing.B) {
	b.RunParallel(func { pb ->
		var t uint64
		for pb.Next() {
			t += Uint64()
		}
		atomic.AddUint64(&Sink, t)
	})
}

func BenchmarkInt64(b *testing.B) {
	r := testRand()
	var t int64
	for n := b.N; n > 0; n-- {
		t += r.Int64()
	}
	Sink = uint64(t)
}

var AlwaysFalse = false

func keep[T int | uint | int32 | uint32 | int64 | uint64](x T) T {
	if AlwaysFalse {
		return -x
	}
	return x
}

func BenchmarkUint64(b *testing.B) {
	r := testRand()
	var t uint64
	for n := b.N; n > 0; n-- {
		t += r.Uint64()
	}
	Sink = t
}

func BenchmarkGlobalIntN1000(b *testing.B) {
	var t int
	arg := keep(1000)
	for n := b.N; n > 0; n-- {
		t += IntN(arg)
	}
	Sink = uint64(t)
}

func BenchmarkIntN1000(b *testing.B) {
	r := testRand()
	var t int
	arg := keep(1000)
	for n := b.N; n > 0; n-- {
		t += r.IntN(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt64N1000(b *testing.B) {
	r := testRand()
	var t int64
	arg := keep(int64(1000))
	for n := b.N; n > 0; n-- {
		t += r.Int64N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt64N1e8(b *testing.B) {
	r := testRand()
	var t int64
	arg := keep(int64(1e8))
	for n := b.N; n > 0; n-- {
		t += r.Int64N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt64N1e9(b *testing.B) {
	r := testRand()
	var t int64
	arg := keep(int64(1e9))
	for n := b.N; n > 0; n-- {
		t += r.Int64N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt64N2e9(b *testing.B) {
	r := testRand()
	var t int64
	arg := keep(int64(2e9))
	for n := b.N; n > 0; n-- {
		t += r.Int64N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt64N1e18(b *testing.B) {
	r := testRand()
	var t int64
	arg := keep(int64(1e18))
	for n := b.N; n > 0; n-- {
		t += r.Int64N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt64N2e18(b *testing.B) {
	r := testRand()
	var t int64
	arg := keep(int64(2e18))
	for n := b.N; n > 0; n-- {
		t += r.Int64N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt64N4e18(b *testing.B) {
	r := testRand()
	var t int64
	arg := keep(int64(4e18))
	for n := b.N; n > 0; n-- {
		t += r.Int64N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt32N1000(b *testing.B) {
	r := testRand()
	var t int32
	arg := keep(int32(1000))
	for n := b.N; n > 0; n-- {
		t += r.Int32N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt32N1e8(b *testing.B) {
	r := testRand()
	var t int32
	arg := keep(int32(1e8))
	for n := b.N; n > 0; n-- {
		t += r.Int32N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt32N1e9(b *testing.B) {
	r := testRand()
	var t int32
	arg := keep(int32(1e9))
	for n := b.N; n > 0; n-- {
		t += r.Int32N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkInt32N2e9(b *testing.B) {
	r := testRand()
	var t int32
	arg := keep(int32(2e9))
	for n := b.N; n > 0; n-- {
		t += r.Int32N(arg)
	}
	Sink = uint64(t)
}

func BenchmarkFloat32(b *testing.B) {
	r := testRand()
	var t float32
	for n := b.N; n > 0; n-- {
		t += r.Float32()
	}
	Sink = uint64(t)
}

func BenchmarkFloat64(b *testing.B) {
	r := testRand()
	var t float64
	for n := b.N; n > 0; n-- {
		t += r.Float64()
	}
	Sink = uint64(t)
}

func BenchmarkExpFloat64(b *testing.B) {
	r := testRand()
	var t float64
	for n := b.N; n > 0; n-- {
		t += r.ExpFloat64()
	}
	Sink = uint64(t)
}

func BenchmarkNormFloat64(b *testing.B) {
	r := testRand()
	var t float64
	for n := b.N; n > 0; n-- {
		t += r.NormFloat64()
	}
	Sink = uint64(t)
}

func BenchmarkPerm3(b *testing.B) {
	r := testRand()
	var t int
	for n := b.N; n > 0; n-- {
		t += r.Perm(3)[0]
	}
	Sink = uint64(t)

}

func BenchmarkPerm30(b *testing.B) {
	r := testRand()
	var t int
	for n := b.N; n > 0; n-- {
		t += r.Perm(30)[0]
	}
	Sink = uint64(t)
}

func BenchmarkPerm30ViaShuffle(b *testing.B) {
	r := testRand()
	var t int
	for n := b.N; n > 0; n-- {
		p := make([]int, 30)
		for i := range p {
			p[i] = i
		}
		r.Shuffle(30, func { i, j -> p[i], p[j] = p[j], p[i] })
		t += p[0]
	}
	Sink = uint64(t)
}

// BenchmarkShuffleOverhead uses a minimal swap function
// to measure just the shuffling overhead.
func BenchmarkShuffleOverhead(b *testing.B) {
	r := testRand()
	for n := b.N; n > 0; n-- {
		r.Shuffle(30, func { i, j -> if i < 0 || i >= 30 || j < 0 || j >= 30 {
			b.Fatalf("bad swap(%d, %d)", i, j)
		} })
	}
}

func BenchmarkConcurrent(b *testing.B) {
	const goroutines = 4
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for n := b.N; n > 0; n-- {
				Int64()
			}
		}()
	}
	wg.Wait()
}

func TestN(t *testing.T) {
	for i := 0; i < 1000; i++ {
		v := N(10)
		if v < 0 || v >= 10 {
			t.Fatalf("N(10) returned %d", v)
		}
	}
}
