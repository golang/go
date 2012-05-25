// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"testing"
)

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// slowPaeth is a slow but simple implementation of the Paeth function.
// It is a straight port of the sample code in the PNG spec, section 9.4.
func slowPaeth(a, b, c uint8) uint8 {
	p := int(a) + int(b) - int(c)
	pa := abs(p - int(a))
	pb := abs(p - int(b))
	pc := abs(p - int(c))
	if pa <= pb && pa <= pc {
		return a
	} else if pb <= pc {
		return b
	}
	return c
}

func TestPaeth(t *testing.T) {
	for a := 0; a < 256; a += 15 {
		for b := 0; b < 256; b += 15 {
			for c := 0; c < 256; c += 15 {
				got := paeth(uint8(a), uint8(b), uint8(c))
				want := slowPaeth(uint8(a), uint8(b), uint8(c))
				if got != want {
					t.Errorf("a, b, c = %d, %d, %d: got %d, want %d", a, b, c, got, want)
				}
			}
		}
	}
}

func BenchmarkPaeth(b *testing.B) {
	for i := 0; i < b.N; i++ {
		paeth(uint8(i>>16), uint8(i>>8), uint8(i))
	}
}
