// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64
// +build riscv64

package testbranch

import (
	"testing"
)

func testBEQZ(a int64) (r bool)
func testBGE(a, b int64) (r bool)
func testBGEU(a, b int64) (r bool)
func testBGEZ(a int64) (r bool)
func testBGT(a, b int64) (r bool)
func testBGTU(a, b int64) (r bool)
func testBGTZ(a int64) (r bool)
func testBLE(a, b int64) (r bool)
func testBLEU(a, b int64) (r bool)
func testBLEZ(a int64) (r bool)
func testBLT(a, b int64) (r bool)
func testBLTU(a, b int64) (r bool)
func testBLTZ(a int64) (r bool)
func testBNEZ(a int64) (r bool)

func testGoBGE(a, b int64) bool  { return a >= b }
func testGoBGEU(a, b int64) bool { return uint64(a) >= uint64(b) }
func testGoBGT(a, b int64) bool  { return a > b }
func testGoBGTU(a, b int64) bool { return uint64(a) > uint64(b) }
func testGoBLE(a, b int64) bool  { return a <= b }
func testGoBLEU(a, b int64) bool { return uint64(a) <= uint64(b) }
func testGoBLT(a, b int64) bool  { return a < b }
func testGoBLTU(a, b int64) bool { return uint64(a) < uint64(b) }

func TestBranchCondition(t *testing.T) {
	tests := []struct {
		ins  string
		a    int64
		b    int64
		fn   func(a, b int64) bool
		goFn func(a, b int64) bool
		want bool
	}{
		{"BGE", 0, 1, testBGE, testGoBGE, false},
		{"BGE", 0, 0, testBGE, testGoBGE, true},
		{"BGE", 0, -1, testBGE, testGoBGE, true},
		{"BGE", -1, 0, testBGE, testGoBGE, false},
		{"BGE", 1, 0, testBGE, testGoBGE, true},
		{"BGEU", 0, 1, testBGEU, testGoBGEU, false},
		{"BGEU", 0, 0, testBGEU, testGoBGEU, true},
		{"BGEU", 0, -1, testBGEU, testGoBGEU, false},
		{"BGEU", -1, 0, testBGEU, testGoBGEU, true},
		{"BGEU", 1, 0, testBGEU, testGoBGEU, true},
		{"BGT", 0, 1, testBGT, testGoBGT, false},
		{"BGT", 0, 0, testBGT, testGoBGT, false},
		{"BGT", 0, -1, testBGT, testGoBGT, true},
		{"BGT", -1, 0, testBGT, testGoBGT, false},
		{"BGT", 1, 0, testBGT, testGoBGT, true},
		{"BGTU", 0, 1, testBGTU, testGoBGTU, false},
		{"BGTU", 0, 0, testBGTU, testGoBGTU, false},
		{"BGTU", 0, -1, testBGTU, testGoBGTU, false},
		{"BGTU", -1, 0, testBGTU, testGoBGTU, true},
		{"BGTU", 1, 0, testBGTU, testGoBGTU, true},
		{"BLE", 0, 1, testBLE, testGoBLE, true},
		{"BLE", 0, 0, testBLE, testGoBLE, true},
		{"BLE", 0, -1, testBLE, testGoBLE, false},
		{"BLE", -1, 0, testBLE, testGoBLE, true},
		{"BLE", 1, 0, testBLE, testGoBLE, false},
		{"BLEU", 0, 1, testBLEU, testGoBLEU, true},
		{"BLEU", 0, 0, testBLEU, testGoBLEU, true},
		{"BLEU", 0, -1, testBLEU, testGoBLEU, true},
		{"BLEU", -1, 0, testBLEU, testGoBLEU, false},
		{"BLEU", 1, 0, testBLEU, testGoBLEU, false},
		{"BLT", 0, 1, testBLT, testGoBLT, true},
		{"BLT", 0, 0, testBLT, testGoBLT, false},
		{"BLT", 0, -1, testBLT, testGoBLT, false},
		{"BLT", -1, 0, testBLT, testGoBLT, true},
		{"BLT", 1, 0, testBLT, testGoBLT, false},
		{"BLTU", 0, 1, testBLTU, testGoBLTU, true},
		{"BLTU", 0, 0, testBLTU, testGoBLTU, false},
		{"BLTU", 0, -1, testBLTU, testGoBLTU, true},
		{"BLTU", -1, 0, testBLTU, testGoBLTU, false},
		{"BLTU", 1, 0, testBLTU, testGoBLTU, false},
	}
	for _, test := range tests {
		t.Run(test.ins, func(t *testing.T) {
			if got := test.fn(test.a, test.b); got != test.want {
				t.Errorf("Assembly %v %v, %v = %v, want %v", test.ins, test.a, test.b, got, test.want)
			}
			if got := test.goFn(test.a, test.b); got != test.want {
				t.Errorf("Go %v %v, %v = %v, want %v", test.ins, test.a, test.b, got, test.want)
			}
		})
	}
}

func TestBranchZero(t *testing.T) {
	tests := []struct {
		ins  string
		a    int64
		fn   func(a int64) bool
		want bool
	}{
		{"BEQZ", -1, testBEQZ, false},
		{"BEQZ", 0, testBEQZ, true},
		{"BEQZ", 1, testBEQZ, false},
		{"BGEZ", -1, testBGEZ, false},
		{"BGEZ", 0, testBGEZ, true},
		{"BGEZ", 1, testBGEZ, true},
		{"BGTZ", -1, testBGTZ, false},
		{"BGTZ", 0, testBGTZ, false},
		{"BGTZ", 1, testBGTZ, true},
		{"BLEZ", -1, testBLEZ, true},
		{"BLEZ", 0, testBLEZ, true},
		{"BLEZ", 1, testBLEZ, false},
		{"BLTZ", -1, testBLTZ, true},
		{"BLTZ", 0, testBLTZ, false},
		{"BLTZ", 1, testBLTZ, false},
		{"BNEZ", -1, testBNEZ, true},
		{"BNEZ", 0, testBNEZ, false},
		{"BNEZ", 1, testBNEZ, true},
	}
	for _, test := range tests {
		t.Run(test.ins, func(t *testing.T) {
			if got := test.fn(test.a); got != test.want {
				t.Errorf("%v %v = %v, want %v", test.ins, test.a, got, test.want)
			}
		})
	}
}
