// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64

package ssa

// This file tests the functions addFlags64 and subFlags64 by comparing their
// results to what the chip calculates.

import (
	"runtime"
	"testing"
)

func TestAddFlagsNative(t *testing.T) {
	var numbers = []int64{
		1, 0, -1,
		2, -2,
		1<<63 - 1, -1 << 63,
	}
	coverage := map[flagConstant]bool{}
	for _, x := range numbers {
		for _, y := range numbers {
			a := addFlags64(x, y)
			b := flagRegister2flagConstant(asmAddFlags(x, y), false)
			if a != b {
				t.Errorf("asmAdd diff: x=%x y=%x got=%s want=%s\n", x, y, a, b)
			}
			coverage[a] = true
		}
	}
	if len(coverage) != 9 { // TODO: can we cover all outputs?
		t.Errorf("coverage too small, got %d want 9", len(coverage))
	}
}

func TestSubFlagsNative(t *testing.T) {
	var numbers = []int64{
		1, 0, -1,
		2, -2,
		1<<63 - 1, -1 << 63,
	}
	coverage := map[flagConstant]bool{}
	for _, x := range numbers {
		for _, y := range numbers {
			a := subFlags64(x, y)
			b := flagRegister2flagConstant(asmSubFlags(x, y), true)
			if a != b {
				t.Errorf("asmSub diff: x=%x y=%x got=%s want=%s\n", x, y, a, b)
			}
			coverage[a] = true
		}
	}
	if len(coverage) != 7 { // TODO: can we cover all outputs?
		t.Errorf("coverage too small, got %d want 7", len(coverage))
	}
}

func TestAndFlagsNative(t *testing.T) {
	var numbers = []int64{
		1, 0, -1,
		2, -2,
		1<<63 - 1, -1 << 63,
	}
	coverage := map[flagConstant]bool{}
	for _, x := range numbers {
		for _, y := range numbers {
			a := logicFlags64(x & y)
			b := flagRegister2flagConstant(asmAndFlags(x, y), false)
			if a != b {
				t.Errorf("asmAnd diff: x=%x y=%x got=%s want=%s\n", x, y, a, b)
			}
			coverage[a] = true
		}
	}
	if len(coverage) != 3 {
		t.Errorf("coverage too small, got %d want 3", len(coverage))
	}
}

func asmAddFlags(x, y int64) int
func asmSubFlags(x, y int64) int
func asmAndFlags(x, y int64) int

func flagRegister2flagConstant(x int, sub bool) flagConstant {
	var fcb flagConstantBuilder
	switch runtime.GOARCH {
	case "amd64":
		fcb.Z = x>>6&1 != 0
		fcb.N = x>>7&1 != 0
		fcb.C = x>>0&1 != 0
		if sub {
			// Convert from amd64-sense to arm-sense
			fcb.C = !fcb.C
		}
		fcb.V = x>>11&1 != 0
	case "arm64":
		fcb.Z = x>>30&1 != 0
		fcb.N = x>>31&1 != 0
		fcb.C = x>>29&1 != 0
		fcb.V = x>>28&1 != 0
	default:
		panic("unsupported architecture: " + runtime.GOARCH)
	}
	return fcb.encode()
}
