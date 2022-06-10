// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testmath

import (
	"math"
	"testing"
	"time"
)

type BenchmarkResults []testing.BenchmarkResult

func (b BenchmarkResults) Weight() float64 {
	var weight int
	for _, r := range b {
		weight += r.N
	}
	return float64(weight)
}

func (b BenchmarkResults) Mean() float64 {
	var dur time.Duration
	for _, r := range b {
		dur += r.T * time.Duration(r.N)
	}
	return float64(dur) / b.Weight()
}

func (b BenchmarkResults) Variance() float64 {
	var num float64
	mean := b.Mean()
	for _, r := range b {
		num += math.Pow(float64(r.T)-mean, 2) * float64(r.N)
	}
	return float64(num) / b.Weight()
}
