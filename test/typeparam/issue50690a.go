// run -gcflags=-G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

// Numeric expresses a type constraint satisfied by any numeric type.
type Numeric interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 |
		~complex64 | ~complex128
}

// Sum returns the sum of the provided arguments.
func Sum[T Numeric](args ...T) T {
	var sum T
	for i := 0; i < len(args); i++ {
		sum += args[i]
	}
	return sum
}

// Ledger is an identifiable, financial record.
type Ledger[T ~string, K Numeric] struct {

	// ID identifies the ledger.
	ID T

	// Amounts is a list of monies associated with this ledger.
	Amounts []K

	// SumFn is a function that can be used to sum the amounts
	// in this ledger.
	SumFn func(...K) K
}

func PrintLedger[
	T ~string,
	K Numeric,
	L ~struct {
		ID      T
		Amounts []K
		SumFn   func(...K) K
	},
](l L) {
	fmt.Printf("%s has a sum of %v\n", l.ID, l.SumFn(l.Amounts...))
}

func main() {
	PrintLedger(Ledger[string, int]{
		ID:      "fake",
		Amounts: []int{1, 2, 3},
		SumFn:   Sum[int],
	})
}
