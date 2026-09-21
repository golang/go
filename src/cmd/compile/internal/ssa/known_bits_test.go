// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"iter"
	"math/bits"
	"testing"
)

// allPossibleValuesRejection has identical behavior to allPossibleValues
// but it is implemented with an obviously correct rejection based algorithm.
// We use it to test that allPossibleValues.
func allPossibleValuesRejection(value, known, max int64) func(yield func(v int64) bool) {
	return func(yield func(v int64) bool) {
		for i := int64(0); i <= max; {
			if i&known == value {
				if !yield(i) {
					return
				}
			}

			next, overflow := bits.Add64(uint64(i), 1, 0)
			if overflow != 0 {
				// exit condition in case the 64th bit is unknown.
				break
			}
			i = int64(next)
		}
	}
}

func TestAllPossibleValues(t *testing.T) {
	// We can't test too much since it scales exponentially with the number of unknown bits.
	const tryMask = int64(0b0111_1111)
	for i := int64(0); uint64(i) <= uint64(tryMask); i++ {
		unknown := ^i
		known := i | ^tryMask

		for value := range allPossibleValuesRejection(0, unknown, tryMask) { // don't use allPossibleValues since it's what we are about to test.
			t.Run(fmt.Sprintf("%v", knownBitsEntry{known: known, value: value}), func(t *testing.T) {
				truth, truthStop := iter.Pull(allPossibleValuesRejection(value, known, tryMask))
				defer truthStop()
				dut, dutStop := iter.Pull(allPossibleValues(value, known))
				defer dutStop()
				for i := int64(0); ; i++ {
					want, wantOk := truth()
					got, gotOk := dut()
					if wantOk != gotOk {
						t.Fatalf("unexpected ok at iteration %d: got %v, want %v", i, gotOk, wantOk)
					}
					if !gotOk {
						break
					}

					if got != want {
						t.Errorf("unexpected value at iteration %d: got %b, want %b", i, uint64(got), uint64(want))
					}
				}
			})
		}
	}
}
