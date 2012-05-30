// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package png

import (
	"bytes"
	"math/rand"
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

// slowFilterPaeth is a slow but simple implementation of func filterPaeth.
func slowFilterPaeth(cdat, pdat []byte, bytesPerPixel int) {
	for i := 0; i < bytesPerPixel; i++ {
		cdat[i] += paeth(0, pdat[i], 0)
	}
	for i := bytesPerPixel; i < len(cdat); i++ {
		cdat[i] += paeth(cdat[i-bytesPerPixel], pdat[i], pdat[i-bytesPerPixel])
	}
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

func TestPaethDecode(t *testing.T) {
	pdat0 := make([]byte, 32)
	pdat1 := make([]byte, 32)
	pdat2 := make([]byte, 32)
	cdat0 := make([]byte, 32)
	cdat1 := make([]byte, 32)
	cdat2 := make([]byte, 32)
	r := rand.New(rand.NewSource(1))
	for bytesPerPixel := 1; bytesPerPixel <= 8; bytesPerPixel++ {
		for i := 0; i < 100; i++ {
			for j := range pdat0 {
				pdat0[j] = uint8(r.Uint32())
				cdat0[j] = uint8(r.Uint32())
			}
			copy(pdat1, pdat0)
			copy(pdat2, pdat0)
			copy(cdat1, cdat0)
			copy(cdat2, cdat0)
			filterPaeth(cdat1, pdat1, bytesPerPixel)
			slowFilterPaeth(cdat2, pdat2, bytesPerPixel)
			if !bytes.Equal(cdat1, cdat2) {
				t.Errorf("bytesPerPixel: %d\npdat0: % x\ncdat0: % x\ngot:   % x\nwant:  % x", bytesPerPixel, pdat0, cdat0, cdat1, cdat2)
				break
			}
		}
	}
}
