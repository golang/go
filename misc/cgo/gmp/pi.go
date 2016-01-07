// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	big "."
	"fmt"
	"runtime"
)

var (
	tmp1  = big.NewInt(0)
	tmp2  = big.NewInt(0)
	numer = big.NewInt(1)
	accum = big.NewInt(0)
	denom = big.NewInt(1)
	ten   = big.NewInt(10)
)

func extractDigit() int64 {
	if big.CmpInt(numer, accum) > 0 {
		return -1
	}
	tmp1.Lsh(numer, 1).Add(tmp1, numer).Add(tmp1, accum)
	big.DivModInt(tmp1, tmp2, tmp1, denom)
	tmp2.Add(tmp2, numer)
	if big.CmpInt(tmp2, denom) >= 0 {
		return -1
	}
	return tmp1.Int64()
}

func nextTerm(k int64) {
	y2 := k*2 + 1
	accum.Add(accum, tmp1.Lsh(numer, 1))
	accum.Mul(accum, tmp1.SetInt64(y2))
	numer.Mul(numer, tmp1.SetInt64(k))
	denom.Mul(denom, tmp1.SetInt64(y2))
}

func eliminateDigit(d int64) {
	accum.Sub(accum, tmp1.Mul(denom, tmp1.SetInt64(d)))
	accum.Mul(accum, ten)
	numer.Mul(numer, ten)
}

func main() {
	i := 0
	k := int64(0)
	for {
		d := int64(-1)
		for d < 0 {
			k++
			nextTerm(k)
			d = extractDigit()
		}
		eliminateDigit(d)
		fmt.Printf("%c", d+'0')

		if i++; i%50 == 0 {
			fmt.Printf("\n")
			if i >= 1000 {
				break
			}
		}
	}

	fmt.Printf("\n%d calls; bit sizes: %d %d %d\n", runtime.NumCgoCall(), numer.Len(), accum.Len(), denom.Len())
}
