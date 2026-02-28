// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jpeg

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

func benchmarkDCT(b *testing.B, f func(*block)) {
	var blk block // avoid potential allocation in loop
	for b.Loop() {
		for _, blk = range testBlocks {
			f(&blk)
		}
	}
}

func BenchmarkFDCT(b *testing.B) {
	benchmarkDCT(b, fdct)
}

func BenchmarkIDCT(b *testing.B) {
	benchmarkDCT(b, idct)
}

const testSlowVsBig = true

func TestDCT(t *testing.T) {
	blocks := make([]block, len(testBlocks))
	copy(blocks, testBlocks[:])

	// All zeros
	blocks = append(blocks, block{})

	// Every possible unit impulse.
	for i := range blockSize {
		var b block
		b[i] = 255
		blocks = append(blocks, b)
	}

	// All ones.
	var ones block
	for i := range ones {
		ones[i] = 255
	}
	blocks = append(blocks, ones)

	// Every possible inverted unit impulse.
	for i := range blockSize {
		ones[i] = 0
		blocks = append(blocks, ones)
		ones[i] = 255
	}

	// Some randomly generated blocks of varying sparseness.
	r := rand.New(rand.NewSource(123))
	for i := 0; i < 100; i++ {
		b := block{}
		n := r.Int() % 64
		for j := 0; j < n; j++ {
			b[r.Int()%len(b)] = r.Int31() % 256
		}
		blocks = append(blocks, b)
	}

	// Check that the slow FDCT and IDCT functions are inverses,
	// after a scale and level shift.
	// Scaling reduces the rounding errors in the conversion.
	// The “fast” ones are not inverses because the fast IDCT
	// is optimized for 8-bit inputs, not full 16-bit ones.
	slowRoundTrip := func(b *block) {
		slowFDCT(b)
		slowIDCT(b)
		for j := range b {
			b[j] = b[j]/8 + 128
		}
	}
	nop := func(*block) {}
	testDCT(t, "IDCT(FDCT)", blocks, slowRoundTrip, nop, 1, 8)

	if testSlowVsBig {
		testDCT(t, "slowFDCT", blocks, slowFDCT, slowerFDCT, 0, 64)
		testDCT(t, "slowIDCT", blocks, slowIDCT, slowerIDCT, 0, 64)
	}

	// Check that the optimized and slow FDCT implementations agree.
	testDCT(t, "FDCT", blocks, fdct, slowFDCT, 1, 8)
	testDCT(t, "IDCT", blocks, idct, slowIDCT, 1, 8)
}

func testDCT(t *testing.T, name string, blocks []block, fhave, fwant func(*block), tolerance int32, maxCloseCalls int) {
	t.Run(name, func(t *testing.T) {
		totalClose := 0
		for i, b := range blocks {
			have, want := b, b
			fhave(&have)
			fwant(&want)
			d, n := differ(&have, &want, tolerance)
			if d >= 0 || n > maxCloseCalls {
				fail := ""
				if d >= 0 {
					fail = fmt.Sprintf("diff at %d,%d", d/8, d%8)
				}
				if n > maxCloseCalls {
					if fail != "" {
						fail += "; "
					}
					fail += fmt.Sprintf("%d close calls", n)
				}
				t.Errorf("i=%d: %s (%s)\nsrc\n%s\nhave\n%s\nwant\n%s\n",
					i, name, fail, &b, &have, &want)
			}
			totalClose += n
		}
		if tolerance > 0 {
			t.Logf("%d/%d total close calls", totalClose, len(blocks)*blockSize)
		}
	})
}

// differ returns the index of the first pair-wise elements in b0 and b1
// that differ by more than 'ok', along with the total number of elements
// that differ by at least ok ("close calls").
//
// There isn't a single definitive decoding of a given JPEG image,
// even before the YCbCr to RGB conversion; implementations
// can have different IDCT rounding errors.
//
// If there are no differences, differ returns -1, 0.
func differ(b0, b1 *block, ok int32) (index, closeCalls int) {
	index = -1
	for i := range b0 {
		delta := b0[i] - b1[i]
		if delta < -ok || ok < delta {
			if index < 0 {
				index = i
			}
		}
		if delta <= -ok || ok <= delta {
			closeCalls++
		}
	}
	return
}

// alpha returns 1 if i is 0 and returns √2 otherwise.
func alpha(i int) float64 {
	if i == 0 {
		return 1
	}
	return math.Sqrt2
}

// bigAlpha returns 1 if i is 0 and returns √2 otherwise.
func bigAlpha(i int) *big.Float {
	if i == 0 {
		return bigFloat1
	}
	return bigFloatSqrt2
}

var cosines = [32]float64{
	+1.0000000000000000000000000000000000000000000000000000000000000000, // cos(π/16 *  0)
	+0.9807852804032304491261822361342390369739337308933360950029160885, // cos(π/16 *  1)
	+0.9238795325112867561281831893967882868224166258636424861150977312, // cos(π/16 *  2)
	+0.8314696123025452370787883776179057567385608119872499634461245902, // cos(π/16 *  3)
	+0.7071067811865475244008443621048490392848359376884740365883398689, // cos(π/16 *  4)
	+0.5555702330196022247428308139485328743749371907548040459241535282, // cos(π/16 *  5)
	+0.3826834323650897717284599840303988667613445624856270414338006356, // cos(π/16 *  6)
	+0.1950903220161282678482848684770222409276916177519548077545020894, // cos(π/16 *  7)

	-0.0000000000000000000000000000000000000000000000000000000000000000, // cos(π/16 *  8)
	-0.1950903220161282678482848684770222409276916177519548077545020894, // cos(π/16 *  9)
	-0.3826834323650897717284599840303988667613445624856270414338006356, // cos(π/16 * 10)
	-0.5555702330196022247428308139485328743749371907548040459241535282, // cos(π/16 * 11)
	-0.7071067811865475244008443621048490392848359376884740365883398689, // cos(π/16 * 12)
	-0.8314696123025452370787883776179057567385608119872499634461245902, // cos(π/16 * 13)
	-0.9238795325112867561281831893967882868224166258636424861150977312, // cos(π/16 * 14)
	-0.9807852804032304491261822361342390369739337308933360950029160885, // cos(π/16 * 15)

	-1.0000000000000000000000000000000000000000000000000000000000000000, // cos(π/16 * 16)
	-0.9807852804032304491261822361342390369739337308933360950029160885, // cos(π/16 * 17)
	-0.9238795325112867561281831893967882868224166258636424861150977312, // cos(π/16 * 18)
	-0.8314696123025452370787883776179057567385608119872499634461245902, // cos(π/16 * 19)
	-0.7071067811865475244008443621048490392848359376884740365883398689, // cos(π/16 * 20)
	-0.5555702330196022247428308139485328743749371907548040459241535282, // cos(π/16 * 21)
	-0.3826834323650897717284599840303988667613445624856270414338006356, // cos(π/16 * 22)
	-0.1950903220161282678482848684770222409276916177519548077545020894, // cos(π/16 * 23)

	+0.0000000000000000000000000000000000000000000000000000000000000000, // cos(π/16 * 24)
	+0.1950903220161282678482848684770222409276916177519548077545020894, // cos(π/16 * 25)
	+0.3826834323650897717284599840303988667613445624856270414338006356, // cos(π/16 * 26)
	+0.5555702330196022247428308139485328743749371907548040459241535282, // cos(π/16 * 27)
	+0.7071067811865475244008443621048490392848359376884740365883398689, // cos(π/16 * 28)
	+0.8314696123025452370787883776179057567385608119872499634461245902, // cos(π/16 * 29)
	+0.9238795325112867561281831893967882868224166258636424861150977312, // cos(π/16 * 30)
	+0.9807852804032304491261822361342390369739337308933360950029160885, // cos(π/16 * 31)
}

func bigFloat(s string) *big.Float {
	f, ok := new(big.Float).SetString(s)
	if !ok {
		panic("bad float")
	}
	return f
}

var (
	bigFloat1     = big.NewFloat(1)
	bigFloatSqrt2 = bigFloat("1.41421356237309504880168872420969807856967187537694807317667974")
)

var bigCosines = [32]*big.Float{
	bigFloat("+1.0000000000000000000000000000000000000000000000000000000000000000"), // cos(π/16 *  0)
	bigFloat("+0.9807852804032304491261822361342390369739337308933360950029160885"), // cos(π/16 *  1)
	bigFloat("+0.9238795325112867561281831893967882868224166258636424861150977312"), // cos(π/16 *  2)
	bigFloat("+0.8314696123025452370787883776179057567385608119872499634461245902"), // cos(π/16 *  3)
	bigFloat("+0.7071067811865475244008443621048490392848359376884740365883398689"), // cos(π/16 *  4)
	bigFloat("+0.5555702330196022247428308139485328743749371907548040459241535282"), // cos(π/16 *  5)
	bigFloat("+0.3826834323650897717284599840303988667613445624856270414338006356"), // cos(π/16 *  6)
	bigFloat("+0.1950903220161282678482848684770222409276916177519548077545020894"), // cos(π/16 *  7)

	bigFloat("-0.0000000000000000000000000000000000000000000000000000000000000000"), // cos(π/16 *  8)
	bigFloat("-0.1950903220161282678482848684770222409276916177519548077545020894"), // cos(π/16 *  9)
	bigFloat("-0.3826834323650897717284599840303988667613445624856270414338006356"), // cos(π/16 * 10)
	bigFloat("-0.5555702330196022247428308139485328743749371907548040459241535282"), // cos(π/16 * 11)
	bigFloat("-0.7071067811865475244008443621048490392848359376884740365883398689"), // cos(π/16 * 12)
	bigFloat("-0.8314696123025452370787883776179057567385608119872499634461245902"), // cos(π/16 * 13)
	bigFloat("-0.9238795325112867561281831893967882868224166258636424861150977312"), // cos(π/16 * 14)
	bigFloat("-0.9807852804032304491261822361342390369739337308933360950029160885"), // cos(π/16 * 15)

	bigFloat("-1.0000000000000000000000000000000000000000000000000000000000000000"), // cos(π/16 * 16)
	bigFloat("-0.9807852804032304491261822361342390369739337308933360950029160885"), // cos(π/16 * 17)
	bigFloat("-0.9238795325112867561281831893967882868224166258636424861150977312"), // cos(π/16 * 18)
	bigFloat("-0.8314696123025452370787883776179057567385608119872499634461245902"), // cos(π/16 * 19)
	bigFloat("-0.7071067811865475244008443621048490392848359376884740365883398689"), // cos(π/16 * 20)
	bigFloat("-0.5555702330196022247428308139485328743749371907548040459241535282"), // cos(π/16 * 21)
	bigFloat("-0.3826834323650897717284599840303988667613445624856270414338006356"), // cos(π/16 * 22)
	bigFloat("-0.1950903220161282678482848684770222409276916177519548077545020894"), // cos(π/16 * 23)

	bigFloat("+0.0000000000000000000000000000000000000000000000000000000000000000"), // cos(π/16 * 24)
	bigFloat("+0.1950903220161282678482848684770222409276916177519548077545020894"), // cos(π/16 * 25)
	bigFloat("+0.3826834323650897717284599840303988667613445624856270414338006356"), // cos(π/16 * 26)
	bigFloat("+0.5555702330196022247428308139485328743749371907548040459241535282"), // cos(π/16 * 27)
	bigFloat("+0.7071067811865475244008443621048490392848359376884740365883398689"), // cos(π/16 * 28)
	bigFloat("+0.8314696123025452370787883776179057567385608119872499634461245902"), // cos(π/16 * 29)
	bigFloat("+0.9238795325112867561281831893967882868224166258636424861150977312"), // cos(π/16 * 30)
	bigFloat("+0.9807852804032304491261822361342390369739337308933360950029160885"), // cos(π/16 * 31)
}

// slowFDCT performs the 8*8 2-dimensional forward discrete cosine transform:
//
//	dst[u,v] = (1/8) * Σ_x Σ_y alpha(u) * alpha(v) * src[x,y] *
//		cos((π/2) * (2*x + 1) * u / 8) *
//		cos((π/2) * (2*y + 1) * v / 8)
//
// x and y are in pixel space, and u and v are in transform space.
//
// b acts as both dst and src.
func slowFDCT(b *block) {
	var dst block
	for v := 0; v < 8; v++ {
		for u := 0; u < 8; u++ {
			sum := 0.0
			for y := 0; y < 8; y++ {
				for x := 0; x < 8; x++ {
					sum += alpha(u) * alpha(v) * float64(b[8*y+x]-128) *
						cosines[((2*x+1)*u)%32] *
						cosines[((2*y+1)*v)%32]
				}
			}
			dst[8*v+u] = int32(math.Round(sum))
		}
	}
	*b = dst
}

// slowerFDCT is slowFDCT but using big.Floats to validate slowFDCT.
func slowerFDCT(b *block) {
	var dst block
	for v := 0; v < 8; v++ {
		for u := 0; u < 8; u++ {
			sum := big.NewFloat(0)
			for y := 0; y < 8; y++ {
				for x := 0; x < 8; x++ {
					f := big.NewFloat(float64(b[8*y+x] - 128))
					f = new(big.Float).Mul(f, bigAlpha(u))
					f = new(big.Float).Mul(f, bigAlpha(v))
					f = new(big.Float).Mul(f, bigCosines[((2*x+1)*u)%32])
					f = new(big.Float).Mul(f, bigCosines[((2*y+1)*v)%32])
					sum = new(big.Float).Add(sum, f)
				}
			}
			// Int64 truncates toward zero, so add ±0.5
			// as needed to round
			if sum.Sign() > 0 {
				sum = new(big.Float).Add(sum, big.NewFloat(+0.5))
			} else {
				sum = new(big.Float).Add(sum, big.NewFloat(-0.5))
			}
			i, _ := sum.Int64()
			dst[8*v+u] = int32(i)
		}
	}
	*b = dst
}

// slowIDCT performs the 8*8 2-dimensional inverse discrete cosine transform:
//
//	dst[x,y] = (1/8) * Σ_u Σ_v alpha(u) * alpha(v) * src[u,v] *
//		cos((π/2) * (2*x + 1) * u / 8) *
//		cos((π/2) * (2*y + 1) * v / 8)
//
// x and y are in pixel space, and u and v are in transform space.
//
// b acts as both dst and src.
func slowIDCT(b *block) {
	var dst block
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			sum := 0.0
			for v := 0; v < 8; v++ {
				for u := 0; u < 8; u++ {
					sum += alpha(u) * alpha(v) * float64(b[8*v+u]) *
						cosines[((2*x+1)*u)%32] *
						cosines[((2*y+1)*v)%32]
				}
			}
			dst[8*y+x] = int32(math.Round(sum / 8))
		}
	}
	*b = dst
}

// slowerIDCT is slowIDCT but using big.Floats to validate slowIDCT.
func slowerIDCT(b *block) {
	var dst block
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			sum := big.NewFloat(0)
			for v := 0; v < 8; v++ {
				for u := 0; u < 8; u++ {
					f := big.NewFloat(float64(b[8*v+u]))
					f = new(big.Float).Mul(f, bigAlpha(u))
					f = new(big.Float).Mul(f, bigAlpha(v))
					f = new(big.Float).Mul(f, bigCosines[((2*x+1)*u)%32])
					f = new(big.Float).Mul(f, bigCosines[((2*y+1)*v)%32])
					f = new(big.Float).Quo(f, big.NewFloat(8))
					sum = new(big.Float).Add(sum, f)
				}
			}
			// Int64 truncates toward zero, so add ±0.5
			// as needed to round
			if sum.Sign() > 0 {
				sum = new(big.Float).Add(sum, big.NewFloat(+0.5))
			} else {
				sum = new(big.Float).Add(sum, big.NewFloat(-0.5))
			}
			i, _ := sum.Int64()
			dst[8*y+x] = int32(i)
		}
	}
	*b = dst
}

func (b *block) String() string {
	s := &strings.Builder{}
	fmt.Fprintf(s, "{\n")
	for y := 0; y < 8; y++ {
		fmt.Fprintf(s, "\t")
		for x := 0; x < 8; x++ {
			fmt.Fprintf(s, "0x%04x, ", uint16(b[8*y+x]))
		}
		fmt.Fprintln(s)
	}
	fmt.Fprintf(s, "}")
	return s.String()
}

// testBlocks are the first 10 pre-IDCT blocks from ../testdata/video-001.jpeg.
var testBlocks = [10]block{
	{
		0x7f, 0xf6, 0x01, 0x07, 0xff, 0x00, 0x00, 0x00,
		0xf5, 0x01, 0xfa, 0x01, 0xfe, 0x00, 0x01, 0x00,
		0x05, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0xff, 0xf8, 0x00, 0x01, 0xff, 0x00, 0x00,
		0x00, 0x01, 0x00, 0x01, 0x00, 0xff, 0xff, 0x00,
		0xff, 0x0c, 0x00, 0x00, 0x00, 0x00, 0xff, 0x01,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x01, 0xff, 0x01, 0x00, 0xfe,
	},
	{
		0x29, 0x07, 0x00, 0xfc, 0x01, 0x01, 0x00, 0x00,
		0x07, 0x00, 0x03, 0x00, 0x01, 0x00, 0xff, 0xff,
		0xff, 0xfd, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x04, 0x00, 0xff, 0x01, 0x00, 0x00,
		0x01, 0x00, 0x01, 0xff, 0x00, 0x00, 0x00, 0x00,
		0x01, 0xfa, 0x01, 0x00, 0x01, 0x00, 0x01, 0xff,
		0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0x02,
	},
	{
		0xc5, 0xfa, 0x01, 0x00, 0x00, 0x01, 0x00, 0xff,
		0x02, 0xff, 0x01, 0x00, 0x01, 0x00, 0xff, 0x00,
		0xff, 0xff, 0x00, 0xff, 0x01, 0x00, 0x00, 0x00,
		0xff, 0x00, 0x01, 0x00, 0x00, 0x00, 0xff, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
		0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	},
	{
		0x86, 0x05, 0x00, 0x02, 0x00, 0x00, 0x01, 0x00,
		0xf2, 0x06, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00,
		0xf6, 0xfa, 0xf9, 0x00, 0xff, 0x01, 0x00, 0x00,
		0xf9, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xff, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00,
		0xff, 0x00, 0x00, 0x01, 0x00, 0xff, 0x01, 0x00,
		0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x01,
		0x00, 0x01, 0xff, 0x01, 0x00, 0xff, 0x00, 0x00,
	},
	{
		0x24, 0xfe, 0x00, 0xff, 0x00, 0xff, 0xff, 0x00,
		0x08, 0xfd, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00,
		0x06, 0x03, 0x03, 0xff, 0x00, 0x00, 0x00, 0x00,
		0x04, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
		0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01,
		0x01, 0x00, 0x01, 0xff, 0x00, 0x01, 0x00, 0x00,
		0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0xff, 0x01,
	},
	{
		0xcd, 0xff, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
		0x03, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
		0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0xff,
	},
	{
		0x81, 0xfe, 0x05, 0xff, 0x01, 0xff, 0x01, 0x00,
		0xef, 0xf9, 0x00, 0xf9, 0x00, 0xff, 0x00, 0xff,
		0x05, 0xf9, 0x00, 0xf8, 0x01, 0xff, 0x01, 0xff,
		0x00, 0xff, 0x07, 0x00, 0x01, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x01,
		0xff, 0x01, 0x01, 0x00, 0xff, 0x00, 0x00, 0x00,
		0x01, 0x01, 0x00, 0xff, 0x00, 0x00, 0x00, 0xff,
	},
	{
		0x28, 0x00, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x0b, 0x02, 0x01, 0x03, 0x00, 0xff, 0x00, 0x01,
		0xfe, 0x02, 0x01, 0x03, 0xff, 0x00, 0x00, 0x00,
		0x01, 0x00, 0xfd, 0x00, 0x01, 0x00, 0xff, 0x00,
		0x01, 0xff, 0x00, 0xff, 0x01, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0xff, 0x01, 0x01, 0x00, 0xff,
		0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0x00, 0x01,
	},
	{
		0xdf, 0xf9, 0xfe, 0x00, 0x03, 0x01, 0xff, 0xff,
		0x04, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
		0xff, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0xfe, 0x01, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0xff, 0x01, 0x00, 0x00, 0x00, 0x01,
		0xff, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
		0x00, 0xff, 0x00, 0xff, 0x01, 0x00, 0x00, 0x01,
		0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
	},
	{
		0x88, 0xfd, 0x00, 0x00, 0xff, 0x00, 0x01, 0xff,
		0xe1, 0x06, 0x06, 0x01, 0xff, 0x00, 0x01, 0x00,
		0x08, 0x00, 0xfa, 0x00, 0xff, 0xff, 0xff, 0xff,
		0x08, 0x01, 0x00, 0xff, 0x01, 0xff, 0x00, 0x00,
		0xf5, 0xff, 0x00, 0x01, 0xff, 0x01, 0x01, 0x00,
		0xff, 0xff, 0x01, 0xff, 0x01, 0x00, 0x01, 0x00,
		0x00, 0x01, 0x01, 0xff, 0x00, 0xff, 0x00, 0x01,
		0x02, 0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0x00,
	},
}
