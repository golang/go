// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64

package testminmax

import (
	"testing"
)

func testMIN1(a int64) (r int64)
func testMIN2(a, b int64) (r int64)
func testMIN3(a, b int64) (r int64)
func testMIN4(a, b int64) (r int64)
func testMAX1(a int64) (r int64)
func testMAX2(a, b int64) (r int64)
func testMAX3(a, b int64) (r int64)
func testMAX4(a, b int64) (r int64)
func testMINU1(a int64) (r int64)
func testMINU2(a, b int64) (r int64)
func testMINU3(a, b int64) (r int64)
func testMINU4(a, b int64) (r int64)
func testMAXU1(a int64) (r int64)
func testMAXU2(a, b int64) (r int64)
func testMAXU3(a, b int64) (r int64)
func testMAXU4(a, b int64) (r int64)

func TestMin(t *testing.T) {
	tests := []struct {
		a    int64
		b    int64
		want int64
	}{
		{1, 2, 1},
		{2, 1, 1},
		{2, 2, 2},
		{1, -1, -1},
		{-1, 1, -1},
	}
	for _, test := range tests {
		if got := testMIN1(test.a); got != test.a {
			t.Errorf("Assembly testMIN1 %v = %v, want %v", test.a, got, test.a)
		}
		if got := testMIN2(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMIN2 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
		if got := testMIN3(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMIN3 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
		if got := testMIN4(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMIN4 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
	}
}

func TestMax(t *testing.T) {
	tests := []struct {
		a    int64
		b    int64
		want int64
	}{
		{1, 2, 2},
		{2, 1, 2},
		{2, 2, 2},
		{1, -1, 1},
		{-1, 1, 1},
	}
	for _, test := range tests {
		if got := testMAX1(test.a); got != test.a {
			t.Errorf("Assembly testMAX1 %v = %v, want %v", test.a, got, test.a)
		}
		if got := testMAX2(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMAX2 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
		if got := testMAX3(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMAX3 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
		if got := testMAX4(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMAX4 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
	}
}

func TestMinU(t *testing.T) {
	tests := []struct {
		a    int64
		b    int64
		want int64
	}{
		{1, 2, 1},
		{2, 1, 1},
		{2, 2, 2},
		{1, -1, 1},
		{-1, 1, 1},
	}
	for _, test := range tests {
		if got := testMINU1(test.a); got != test.a {
			t.Errorf("Assembly testMINU1 %v = %v, want %v", test.a, got, test.a)
		}
		if got := testMINU2(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMINU2 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
		if got := testMINU3(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMINU3 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
		if got := testMINU4(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMINU4 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
	}
}

func TestMaxU(t *testing.T) {
	tests := []struct {
		a    int64
		b    int64
		want int64
	}{
		{1, 2, 2},
		{2, 1, 2},
		{2, 2, 2},
		{1, -1, -1},
		{-1, 1, -1},
	}
	for _, test := range tests {
		if got := testMAXU1(test.a); got != test.a {
			t.Errorf("Assembly testMAXU1 %v = %v, want %v", test.a, got, test.a)
		}
		if got := testMAXU2(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMAXU2 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
		if got := testMAXU3(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMAXU3 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
		if got := testMAXU4(test.a, test.b); got != test.want {
			t.Errorf("Assembly testMAXU4 %v, %v = %v, want %v", test.a, test.b, got, test.want)
		}
	}
}
