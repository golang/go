// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testmath

import (
	"errors"
	"math"
)

// A TTestSample is a sample that can be used for a one or two sample
// t-test.
type TTestSample interface {
	Weight() float64
	Mean() float64
	Variance() float64
}

var (
	ErrSampleSize        = errors.New("sample is too small")
	ErrZeroVariance      = errors.New("sample has zero variance")
	ErrMismatchedSamples = errors.New("samples have different lengths")
)

// TwoSampleWelchTTest performs a two-sample (unpaired) Welch's t-test
// on samples x1 and x2. This t-test does not assume the distributions
// have equal variance.
func TwoSampleWelchTTest(x1, x2 TTestSample, alt LocationHypothesis) (*TTestResult, error) {
	n1, n2 := x1.Weight(), x2.Weight()
	if n1 <= 1 || n2 <= 1 {
		// TODO: Can we still do this with n == 1?
		return nil, ErrSampleSize
	}
	v1, v2 := x1.Variance(), x2.Variance()
	if v1 == 0 && v2 == 0 {
		return nil, ErrZeroVariance
	}

	dof := math.Pow(v1/n1+v2/n2, 2) /
		(math.Pow(v1/n1, 2)/(n1-1) + math.Pow(v2/n2, 2)/(n2-1))
	s := math.Sqrt(v1/n1 + v2/n2)
	t := (x1.Mean() - x2.Mean()) / s
	return newTTestResult(int(n1), int(n2), t, dof, alt), nil
}

// A TTestResult is the result of a t-test.
type TTestResult struct {
	// N1 and N2 are the sizes of the input samples. For a
	// one-sample t-test, N2 is 0.
	N1, N2 int

	// T is the value of the t-statistic for this t-test.
	T float64

	// DoF is the degrees of freedom for this t-test.
	DoF float64

	// AltHypothesis specifies the alternative hypothesis tested
	// by this test against the null hypothesis that there is no
	// difference in the means of the samples.
	AltHypothesis LocationHypothesis

	// P is p-value for this t-test for the given null hypothesis.
	P float64
}

func newTTestResult(n1, n2 int, t, dof float64, alt LocationHypothesis) *TTestResult {
	dist := TDist{dof}
	var p float64
	switch alt {
	case LocationDiffers:
		p = 2 * (1 - dist.CDF(math.Abs(t)))
	case LocationLess:
		p = dist.CDF(t)
	case LocationGreater:
		p = 1 - dist.CDF(t)
	}
	return &TTestResult{N1: n1, N2: n2, T: t, DoF: dof, AltHypothesis: alt, P: p}
}

// A LocationHypothesis specifies the alternative hypothesis of a
// location test such as a t-test or a Mann-Whitney U-test. The
// default (zero) value is to test against the alternative hypothesis
// that they differ.
type LocationHypothesis int

const (
	// LocationLess specifies the alternative hypothesis that the
	// location of the first sample is less than the second. This
	// is a one-tailed test.
	LocationLess LocationHypothesis = -1

	// LocationDiffers specifies the alternative hypothesis that
	// the locations of the two samples are not equal. This is a
	// two-tailed test.
	LocationDiffers LocationHypothesis = 0

	// LocationGreater specifies the alternative hypothesis that
	// the location of the first sample is greater than the
	// second. This is a one-tailed test.
	LocationGreater LocationHypothesis = 1
)

// A TDist is a Student's t-distribution with V degrees of freedom.
type TDist struct {
	V float64
}

// PDF returns the value at x of the probability distribution function for the
// distribution.
func (t TDist) PDF(x float64) float64 {
	return math.Exp(lgamma((t.V+1)/2)-lgamma(t.V/2)) /
		math.Sqrt(t.V*math.Pi) * math.Pow(1+(x*x)/t.V, -(t.V+1)/2)
}

// CDF returns the value at x of the cumulative distribution function for the
// distribution.
func (t TDist) CDF(x float64) float64 {
	if x == 0 {
		return 0.5
	} else if x > 0 {
		return 1 - 0.5*betaInc(t.V/(t.V+x*x), t.V/2, 0.5)
	} else if x < 0 {
		return 1 - t.CDF(-x)
	} else {
		return math.NaN()
	}
}

func (t TDist) Bounds() (float64, float64) {
	return -4, 4
}

func lgamma(x float64) float64 {
	y, _ := math.Lgamma(x)
	return y
}

// betaInc returns the value of the regularized incomplete beta
// function Iₓ(a, b) = 1 / B(a, b) * ∫₀ˣ tᵃ⁻¹ (1-t)ᵇ⁻¹ dt.
//
// This is not to be confused with the "incomplete beta function",
// which can be computed as BetaInc(x, a, b)*Beta(a, b).
//
// If x < 0 or x > 1, returns NaN.
func betaInc(x, a, b float64) float64 {
	// Based on Numerical Recipes in C, section 6.4. This uses the
	// continued fraction definition of I:
	//
	//  (xᵃ*(1-x)ᵇ)/(a*B(a,b)) * (1/(1+(d₁/(1+(d₂/(1+...))))))
	//
	// where B(a,b) is the beta function and
	//
	//  d_{2m+1} = -(a+m)(a+b+m)x/((a+2m)(a+2m+1))
	//  d_{2m}   = m(b-m)x/((a+2m-1)(a+2m))
	if x < 0 || x > 1 {
		return math.NaN()
	}
	bt := 0.0
	if 0 < x && x < 1 {
		// Compute the coefficient before the continued
		// fraction.
		bt = math.Exp(lgamma(a+b) - lgamma(a) - lgamma(b) +
			a*math.Log(x) + b*math.Log(1-x))
	}
	if x < (a+1)/(a+b+2) {
		// Compute continued fraction directly.
		return bt * betacf(x, a, b) / a
	} else {
		// Compute continued fraction after symmetry transform.
		return 1 - bt*betacf(1-x, b, a)/b
	}
}

// betacf is the continued fraction component of the regularized
// incomplete beta function Iₓ(a, b).
func betacf(x, a, b float64) float64 {
	const maxIterations = 200
	const epsilon = 3e-14

	raiseZero := func(z float64) float64 {
		if math.Abs(z) < math.SmallestNonzeroFloat64 {
			return math.SmallestNonzeroFloat64
		}
		return z
	}

	c := 1.0
	d := 1 / raiseZero(1-(a+b)*x/(a+1))
	h := d
	for m := 1; m <= maxIterations; m++ {
		mf := float64(m)

		// Even step of the recurrence.
		numer := mf * (b - mf) * x / ((a + 2*mf - 1) * (a + 2*mf))
		d = 1 / raiseZero(1+numer*d)
		c = raiseZero(1 + numer/c)
		h *= d * c

		// Odd step of the recurrence.
		numer = -(a + mf) * (a + b + mf) * x / ((a + 2*mf) * (a + 2*mf + 1))
		d = 1 / raiseZero(1+numer*d)
		c = raiseZero(1 + numer/c)
		hfac := d * c
		h *= hfac

		if math.Abs(hfac-1) < epsilon {
			return h
		}
	}
	panic("betainc: a or b too big; failed to converge")
}
