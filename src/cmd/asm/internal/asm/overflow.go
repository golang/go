// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

/*
	Tested with uint8s like this:

	for a := 0; a <= 255; a++ {
		for b := 0; b <= 127; b++ {
			ovfl := a+b != int(uint8(a)+uint8(b))
			if addOverflows(uint8(a), uint8(b)) != ovfl {
				fmt.Printf("%d+%d fails\n", a, b)
				break
			}
		}
	}
	for a := 0; a <= 255; a++ {
		for b := 0; b <= 127; b++ {
			ovfl := a-b != int(uint8(a)-uint8(b))
			if subOverflows(uint8(a), uint8(b)) != ovfl {
				fmt.Printf("%d-%d fails\n", a, b)
				break
			}
		}
	}
	for a := 0; a <= 255; a++ {
		for b := 0; b <= 255; b++ {
			ovfl := a*b != int(uint8(a)*uint8(b))
			if mulOverflows(uint8(a), uint8(b)) != ovfl {
				fmt.Printf("%d*%d fails\n", a, b)
			}
		}
	}
*/

func addOverflows(a, b uint64) bool {
	return a+b < a
}

func subOverflows(a, b uint64) bool {
	return a-b > a
}

func mulOverflows(a, b uint64) bool {
	if a <= 1 || b <= 1 {
		return false
	}
	c := a * b
	return c/b != a
}

/*
For the record, signed overflow:

const mostNegative = -(mostPositive + 1)
const mostPositive = 1<<63 - 1

func signedAddOverflows(a, b int64) bool {
	if (a >= 0) != (b >= 0) {
		// Different signs cannot overflow.
		return false
	}
	if a >= 0 {
		// Both are positive.
		return a+b < 0
	}
	return a+b >= 0
}

func signedSubOverflows(a, b int64) bool {
	if (a >= 0) == (b >= 0) {
		// Same signs cannot overflow.
		return false
	}
	if a >= 0 {
		// a positive, b negative.
		return a-b < 0
	}
	return a-b >= 0
}

func signedMulOverflows(a, b int64) bool {
	if a == 0 || b == 0 || a == 1 || b == 1 {
		return false
	}
	if a == mostNegative || b == mostNegative {
		return true
	}
	c := a * b
	return c/b != a
}
*/
