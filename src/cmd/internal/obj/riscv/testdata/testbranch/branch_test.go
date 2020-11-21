// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

func TestBranchCondition(t *testing.T) {
	tests := []struct {
		ins  string
		a    int64
		b    int64
		fn   func(a, b int64) bool
		want bool
	}{
		{"BGE", 0, 1, testBGE, false},
		{"BGE", 0, 0, testBGE, true},
		{"BGE", 0, -1, testBGE, true},
		{"BGE", -1, 0, testBGE, false},
		{"BGE", 1, 0, testBGE, true},
		{"BGEU", 0, 1, testBGEU, false},
		{"BGEU", 0, 0, testBGEU, true},
		{"BGEU", 0, -1, testBGEU, false},
		{"BGEU", -1, 0, testBGEU, true},
		{"BGEU", 1, 0, testBGEU, true},
		{"BGT", 0, 1, testBGT, false},
		{"BGT", 0, 0, testBGT, false},
		{"BGT", 0, -1, testBGT, true},
		{"BGT", -1, 0, testBGT, false},
		{"BGT", 1, 0, testBGT, true},
		{"BGTU", 0, 1, testBGTU, false},
		{"BGTU", 0, 0, testBGTU, false},
		{"BGTU", 0, -1, testBGTU, false},
		{"BGTU", -1, 0, testBGTU, true},
		{"BGTU", 1, 0, testBGTU, true},
		{"BLE", 0, 1, testBLE, true},
		{"BLE", 0, 0, testBLE, true},
		{"BLE", 0, -1, testBLE, false},
		{"BLE", -1, 0, testBLE, true},
		{"BLE", 1, 0, testBLE, false},
		{"BLEU", 0, 1, testBLEU, true},
		{"BLEU", 0, 0, testBLEU, true},
		{"BLEU", 0, -1, testBLEU, true},
		{"BLEU", -1, 0, testBLEU, false},
		{"BLEU", 1, 0, testBLEU, false},
		{"BLT", 0, 1, testBLT, true},
		{"BLT", 0, 0, testBLT, false},
		{"BLT", 0, -1, testBLT, false},
		{"BLT", -1, 0, testBLT, true},
		{"BLT", 1, 0, testBLT, false},
		{"BLTU", 0, 1, testBLTU, true},
		{"BLTU", 0, 0, testBLTU, false},
		{"BLTU", 0, -1, testBLTU, true},
		{"BLTU", -1, 0, testBLTU, false},
		{"BLTU", 1, 0, testBLTU, false},
	}
	for _, test := range tests {
		t.Run(test.ins, func(t *testing.T) {
			var fn func(a, b int64) bool
			switch test.ins {
			case "BGE":
				fn = func(a, b int64) bool { return a >= b }
			case "BGEU":
				fn = func(a, b int64) bool { return uint64(a) >= uint64(b) }
			case "BGT":
				fn = func(a, b int64) bool { return a > b }
			case "BGTU":
				fn = func(a, b int64) bool { return uint64(a) > uint64(b) }
			case "BLE":
				fn = func(a, b int64) bool { return a <= b }
			case "BLEU":
				fn = func(a, b int64) bool { return uint64(a) <= uint64(b) }
			case "BLT":
				fn = func(a, b int64) bool { return a < b }
			case "BLTU":
				fn = func(a, b int64) bool { return uint64(a) < uint64(b) }
			default:
				t.Fatalf("Unknown instruction %q", test.ins)
			}
			if got := fn(test.a, test.b); got != test.want {
				t.Errorf("Go %v %v, %v = %v, want %v", test.ins, test.a, test.b, got, test.want)
			}
			if got := test.fn(test.a, test.b); got != test.want {
				t.Errorf("Assembly %v %v, %v = %v, want %v", test.ins, test.a, test.b, got, test.want)
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
