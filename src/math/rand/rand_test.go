// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
)

const (
	numTestSamples = 10000
)

type statsResults struct {
	mean        float64
	stddev      float64
	closeEnough float64
	maxError    float64
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func nearEqual(a, b, closeEnough, maxError float64) bool {
	absDiff := math.Abs(a - b)
	if absDiff < closeEnough { // Necessary when one value is zero and one value is close to zero.
		return true
	}
	return absDiff/max(math.Abs(a), math.Abs(b)) < maxError
}

var testSeeds = []int64{1, 1754801282, 1698661970, 1550503961}

// checkSimilarDistribution returns success if the mean and stddev of the
// two statsResults are similar.
func (this *statsResults) checkSimilarDistribution(expected *statsResults) error {
	if !nearEqual(this.mean, expected.mean, expected.closeEnough, expected.maxError) {
		s := fmt.Sprintf("mean %v != %v (allowed error %v, %v)", this.mean, expected.mean, expected.closeEnough, expected.maxError)
		fmt.Println(s)
		return errors.New(s)
	}
	if !nearEqual(this.stddev, expected.stddev, 0, expected.maxError) {
		s := fmt.Sprintf("stddev %v != %v (allowed error %v, %v)", this.stddev, expected.stddev, expected.closeEnough, expected.maxError)
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
	actual := getStatsResults(samples)
	err := actual.checkSimilarDistribution(expected)
	if err != nil {
		t.Errorf(err.Error())
	}
}

func checkSampleSliceDistributions(t *testing.T, samples []float64, nslices int, expected *statsResults) {
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

func generateNormalSamples(nsamples int, mean, stddev float64, seed int64) []float64 {
	r := New(NewSource(seed))
	samples := make([]float64, nsamples)
	for i := range samples {
		samples[i] = r.NormFloat64()*stddev + mean
	}
	return samples
}

func testNormalDistribution(t *testing.T, nsamples int, mean, stddev float64, seed int64) {
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

func generateExponentialSamples(nsamples int, rate float64, seed int64) []float64 {
	r := New(NewSource(seed))
	samples := make([]float64, nsamples)
	for i := range samples {
		samples[i] = r.ExpFloat64() / rate
	}
	return samples
}

func testExponentialDistribution(t *testing.T, nsamples int, rate float64, seed int64) {
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

func TestFloat32(t *testing.T) {
	// For issue 6721, the problem came after 7533753 calls, so check 10e6.
	num := int(10e6)
	// But ARM5 floating point emulation is slow (Issue 10749), so
	// do less for that builder:
	if testing.Short() && runtime.GOARCH == "arm" && os.Getenv("GOARM") == "5" {
		num /= 100 // 1.72 seconds instead of 172 seconds
	}

	r := New(NewSource(1))
	for ct := 0; ct < num; ct++ {
		f := r.Float32()
		if f >= 1 {
			t.Fatal("Float32() should be in range [0,1). ct:", ct, "f:", f)
		}
	}
}

// Benchmarks

func BenchmarkInt63Threadsafe(b *testing.B) {
	for n := b.N; n > 0; n-- {
		Int63()
	}
}

func BenchmarkInt63Unthreadsafe(b *testing.B) {
	r := New(NewSource(1))
	for n := b.N; n > 0; n-- {
		r.Int63()
	}
}

func BenchmarkIntn1000(b *testing.B) {
	r := New(NewSource(1))
	for n := b.N; n > 0; n-- {
		r.Intn(1000)
	}
}

func BenchmarkInt63n1000(b *testing.B) {
	r := New(NewSource(1))
	for n := b.N; n > 0; n-- {
		r.Int63n(1000)
	}
}

func BenchmarkInt31n1000(b *testing.B) {
	r := New(NewSource(1))
	for n := b.N; n > 0; n-- {
		r.Int31n(1000)
	}
}

func BenchmarkFloat32(b *testing.B) {
	r := New(NewSource(1))
	for n := b.N; n > 0; n-- {
		r.Float32()
	}
}

func BenchmarkFloat64(b *testing.B) {
	r := New(NewSource(1))
	for n := b.N; n > 0; n-- {
		r.Float64()
	}
}

func BenchmarkPerm3(b *testing.B) {
	r := New(NewSource(1))
	for n := b.N; n > 0; n-- {
		r.Perm(3)
	}
}

func BenchmarkPerm30(b *testing.B) {
	r := New(NewSource(1))
	for n := b.N; n > 0; n-- {
		r.Perm(30)
	}
}
