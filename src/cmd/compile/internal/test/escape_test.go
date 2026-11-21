// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package test

import (
	"internal/testenv"
	"testing"
)

type largeLitElem struct {
	a, b int
}

//go:noinline
func consumeLargeLitPtrs(s []*largeLitElem) {}

//go:noinline
func largeSliceLitWithNew() {
	s := []*largeLitElem{(1<<17 - 1): new(largeLitElem)}[:0]
	consumeLargeLitPtrs(s)
}

//go:noinline
func largeSliceLitWithTwoNews() {
	s := []*largeLitElem{
		0:           new(largeLitElem),
		(1<<17 - 1): new(largeLitElem),
	}[:0]
	consumeLargeLitPtrs(s)
}

func TestLargeSliceLiteralPointeesOnHeap(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)

	for _, tc := range []struct {
		name string
		f    func()
		want float64
	}{
		{"OneNew", largeSliceLitWithNew, 2},
		{"TwoNews", largeSliceLitWithTwoNews, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := testing.AllocsPerRun(10, tc.f); got != tc.want {
				t.Errorf("got %v allocs, want %v", got, tc.want)
			}
		})
	}
}
