// run -race

//go:build cgo && linux && amd64

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type LineString []Point
type Point [2]float64

//go:noinline
func benchmarkData() LineString {
	return LineString{{1.0, 2.0}}
}

func (ls LineString) Clone() LineString {
	ps := MultiPoint(ls)
	return LineString(ps.Clone())
}

type MultiPoint []Point

func (mp MultiPoint) Clone() MultiPoint {
	if mp == nil {
		return nil
	}

	points := make([]Point, len(mp))
	copy(points, mp)

	return MultiPoint(points)
}

func F1() {
	cases := []struct {
		threshold float64
		length    int
	}{
		{0.1, 1118},
		{0.5, 257},
		{1.0, 144},
		{1.5, 95},
		{2.0, 71},
		{3.0, 46},
		{4.0, 39},
		{5.0, 33},
	}

	ls := benchmarkData()

	for k := 0; k < 100; k++ {
		for i, tc := range cases {
			r := DouglasPeucker(tc.threshold).LineString(ls.Clone())
			if len(r) == tc.length {
				fmt.Printf("%d: unexpected\n", i)
			}
		}
	}
}

// A DouglasPeuckerSimplifier wraps the DouglasPeucker function.
type DouglasPeuckerSimplifier struct {
	Threshold float64
}

// DouglasPeucker creates a new DouglasPeuckerSimplifier.
func DouglasPeucker(threshold float64) *DouglasPeuckerSimplifier {
	return &DouglasPeuckerSimplifier{
		Threshold: threshold,
	}
}

func (s *DouglasPeuckerSimplifier) LineString(ls LineString) LineString {
	return lineString(s, ls)
}

type simplifier interface {
	simplify(LineString, bool) (LineString, []int)
}

func lineString(s simplifier, ls LineString) LineString {
	return runSimplify(s, ls)
}

func runSimplify(s simplifier, ls LineString) LineString {
	if len(ls) <= 2 {
		return ls
	}
	ls, _ = s.simplify(ls, false)
	return ls
}

func (s *DouglasPeuckerSimplifier) simplify(ls LineString, wim bool) (LineString, []int) {
	return nil, nil
}

func main() {
	F1()
}
