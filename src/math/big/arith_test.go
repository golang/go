// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines tests of consistent behavior between assembly and Go versions of basic operators,
// as well as tests of pure Go implementations.

package big

import (
	"fmt"
	"internal/testenv"
	"iter"
	"math/bits"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

var isRaceBuilder = strings.HasSuffix(testenv.Builder(), "-race")

var words4 = []Word{0, 1, _M - 1, _M}
var words2 = []Word{0, _M}
var muls = []Word{0, 1, 2, 3, 4, 5, _M / 4, _M / 2, _M - 3, _M - 2, _M - 1, _M}
var adds = []Word{0, 1, _M - 1, _M}
var shifts = []uint{1, 2, 3, _W/4 - 1, _W / 4, _W/4 + 1, _W/2 - 1, _W / 2, _W/2 + 1, _W - 3, _W - 2, _W - 1}

func TestAddVV(t *testing.T)      { testVV(t, "addVV", addVV, addVV_g) }
func TestSubVV(t *testing.T)      { testVV(t, "subVV", subVV, subVV_g) }
func TestAddVW(t *testing.T)      { testVW(t, "addVW", addVW, addVW_ref, words4) }
func TestSubVW(t *testing.T)      { testVW(t, "subVW", subVW, subVW_ref, words4) }
func TestLshVU(t *testing.T)      { testVU(t, "lshVU", lshVU, lshVU_g, shifts) }
func TestRshVU(t *testing.T)      { testVU(t, "rshVU", rshVU, rshVU_g, shifts) }
func TestMulAddVWW(t *testing.T)  { testVWW(t, "mulAddVWW", mulAddVWW, mulAddVWW_g, muls) }
func TestAddMulVVWW(t *testing.T) { testVVWW(t, "addMulVVWW", addMulVVWW, addMulVVWW_g, muls, adds) }

// Note: It would be nice to avoid all the duplication of these test variants,
// but the only obvious way is to use reflection. These tests are already
// pretty expensive, and hitting them with reflect call overhead would
// reduce the amount of exhaustive testing it's reasonable to do, so instead
// we put up with the duplication.

func testVV(t *testing.T, name string, fn, ref func(z, x, y []Word) (c Word)) {
	for size := range 100 {
		xx := make([]Word, 1+size+1)
		yy := make([]Word, 1+size+1)
		zz := make([]Word, 1+size+1)
		words := words4
		if size > 5 {
			words = words2
		}
		if size > 10 {
			words = nil // random
		}
		for x := range nats(words, size) {
			for y := range nats(words, size) {
				wantZ := make([]Word, size)
				wantC := ref(wantZ, x, y)

				for _, inplace := range []bool{false, true} {
					name := name
					if inplace {
						name = "in-place " + name
					}
					setSlice(xx, 1, x)
					setSlice(yy, 2, y)
					zz := zz
					if inplace {
						zz = xx
					} else {
						for i := range zz {
							zz[i] = 0x9876
						}
					}
					setSlice(zz, 3, nil)
					c := fn(zz[1:1+size], xx[1:1+size], yy[1:1+size])
					if !slices.Equal(zz[1:1+size], wantZ) || c != wantC {
						t.Errorf("%s(%#x, %#x) = %#x, %#x, want %#x, %#x", name, x, y, zz[1:1+size], c, wantZ, wantC)
					}
					if !inplace {
						checkSlice(t, name, "x", xx, 1, x)
					}
					checkSlice(t, name, "y", yy, 2, y)
					checkSlice(t, name, "z", zz, 3, nil)
					if t.Failed() {
						t.FailNow()
					}
				}
			}
		}
	}
}

func testVV2(t *testing.T, name string, fn, ref func(z1, z2, x, y []Word) (c1, c2 Word)) {
	for size := range 100 {
		xx := make([]Word, 1+size+1)
		yy := make([]Word, 1+size+1)
		zz1 := make([]Word, 1+size+1)
		zz2 := make([]Word, 1+size+1)
		words := words4
		if size > 5 {
			words = words2
		}
		if size > 10 {
			words = nil // random
		}
		for x := range nats(words, size) {
			for y := range nats(words, size) {
				wantZ1 := make([]Word, size)
				wantZ2 := make([]Word, size)
				wantC1, wantC2 := ref(wantZ1, wantZ2, x, y)

				for _, inplace := range []bool{false, true} {
					name := name
					if inplace {
						name = "in-place " + name
					}
					setSlice(xx, 1, x)
					setSlice(yy, 2, y)
					zz1 := zz1
					zz2 := zz2
					if inplace {
						zz1 = xx
						zz2 = yy
					} else {
						for i := range zz1 {
							zz1[i] = 0x9876
						}
						for i := range zz2 {
							zz2[i] = 0x8765
						}
					}
					setSlice(zz1, 3, nil)
					setSlice(zz2, 4, nil)
					c1, c2 := fn(zz1[1:1+size], zz2[1:1+size], xx[1:1+size], yy[1:1+size])
					if !slices.Equal(zz1[1:1+size], wantZ1) || !slices.Equal(zz2[1:1+size], wantZ2) || c1 != wantC1 || c2 != wantC2 {
						t.Errorf("%s(%#x, %#x) = %#x, %#x, %#x, %#x, want %#x, %#x, %#x, %#x", name, x, y, zz1[1:1+size], zz2[1:1+size], c1, c2, wantZ1, wantZ2, wantC1, wantC2)
					}
					if !inplace {
						checkSlice(t, name, "x", xx, 1, x)
						checkSlice(t, name, "y", yy, 2, y)
					}
					checkSlice(t, name, "z1", zz1, 3, nil)
					checkSlice(t, name, "z2", zz2, 4, nil)
					if t.Failed() {
						t.FailNow()
					}
				}
			}
		}
	}
}

func testVW(t *testing.T, name string, fn, ref func(z, x []Word, w Word) (c Word), ws []Word) {
	const (
		magic0 = 0x123450
		magic1 = 0x543210
	)

	for size := range 100 {
		xx := make([]Word, 1+size+1)
		zz := make([]Word, 1+size+1)
		words := words4
		if size > 5 {
			words = words2
		}
		if size > 10 {
			words = nil // random
		}
		for x := range nats(words, size) {
			for _, w := range ws {
				wantZ := make([]Word, size)
				wantC := ref(wantZ, x, w)

				copy(xx[1:], x)
				for _, inplace := range []bool{false, true} {
					name := name
					if inplace {
						name = "in-place " + name
					}
					setSlice(xx, 1, x)
					zz := zz
					if inplace {
						zz = xx
					} else {
						for i := range zz {
							zz[i] = 0x9876
						}
					}
					setSlice(zz, 2, nil)
					c := fn(zz[1:1+size], xx[1:1+size], w)
					if !slices.Equal(zz[1:1+size], wantZ) || c != wantC {
						t.Errorf("%s(%#x, %#x) = %#x, %#x, want %#x, %#x", name, x, w, zz[1:1+size], c, wantZ, wantC)
					}
					if !inplace {
						checkSlice(t, name, "x", xx, 1, x)
					}
					checkSlice(t, name, "z", zz, 2, nil)
					if t.Failed() {
						t.FailNow()
					}
				}
			}
		}
	}
}

func testVU(t *testing.T, name string, fn, ref func(z, x []Word, y uint) (c Word), ys []uint) {
	wys := make([]Word, len(ys))
	for i, y := range ys {
		wys[i] = Word(y)
	}
	testVW(t, name,
		func(z, x []Word, y Word) Word { return fn(z, x, uint(y)) },
		func(z, x []Word, y Word) Word { return ref(z, x, uint(y)) },
		wys)
}

func testVWW(t *testing.T, name string, fn, ref func(z, x []Word, y, r Word) (c Word), ys []Word) {
	const (
		magic0 = 0x123450
		magic1 = 0x543210
	)

	for size := range 100 {
		xx := make([]Word, 1+size+1)
		zz := make([]Word, 1+size+1)
		words := words4
		if size > 5 {
			words = words2
		}
		if size > 10 {
			words = nil // random
		}
		for x := range nats(words, size) {
			for _, y := range ys {
				for _, r := range ys {
					wantZ := make([]Word, size)
					wantC := ref(wantZ, x, y, r)

					copy(xx[1:], x)
					for _, inplace := range []bool{false, true} {
						name := name
						if inplace {
							name = "in-place " + name
						}
						setSlice(xx, 1, x)
						zz := zz
						if inplace {
							zz = xx
						} else {
							for i := range zz {
								zz[i] = 0x9876
							}
						}
						setSlice(zz, 2, nil)
						c := fn(zz[1:1+size], xx[1:1+size], y, r)
						if !slices.Equal(zz[1:1+size], wantZ) || c != wantC {
							t.Errorf("%s(%#x, %#x, %#x) = %#x, %#x, want %#x, %#x", name, x, y, r, zz[1:1+size], c, wantZ, wantC)
						}
						if !inplace {
							checkSlice(t, name, "x", xx, 1, x)
						}
						checkSlice(t, name, "z", zz, 2, nil)
						if t.Failed() {
							t.FailNow()
						}
					}
				}
			}
		}
	}
}

func testVVU(t *testing.T, name string, fn, ref func(z, x, y []Word, s uint) (c Word), shifts []uint) {
	for size := range 100 {
		xx := make([]Word, 1+size+1)
		yy := make([]Word, 1+size+1)
		zz := make([]Word, 1+size+1)
		words := words4
		if size > 5 {
			words = words2
		}
		if size > 10 {
			words = nil // random
		}
		for x := range nats(words, size) {
			for y := range nats(words, size) {
				for _, s := range shifts {
					wantZ := make([]Word, size)
					wantC := ref(wantZ, x, y, s)

					for _, inplace := range []bool{false, true} {
						name := name
						if inplace {
							name = "in-place " + name
						}
						setSlice(xx, 1, x)
						setSlice(yy, 2, y)
						zz := zz
						if inplace {
							zz = xx
						} else {
							for i := range zz {
								zz[i] = 0x9876
							}
						}
						setSlice(zz, 3, nil)
						c := fn(zz[1:1+size], xx[1:1+size], yy[1:1+size], s)
						if !slices.Equal(zz[1:1+size], wantZ) || c != wantC {
							t.Errorf("%s(%#x, %#x, %#x) = %#x, %#x, want %#x, %#x", name, x, y, s, zz[1:1+size], c, wantZ, wantC)
						}
						if !inplace {
							checkSlice(t, name, "x", xx, 1, x)
						}
						checkSlice(t, name, "y", yy, 2, y)
						checkSlice(t, name, "z", zz, 3, nil)
						if t.Failed() {
							t.FailNow()
						}
					}
				}
			}
		}
	}
}

func testVVWW(t *testing.T, name string, fn, ref func(z, x, y []Word, m, a Word) (c Word), ms, as []Word) {
	for size := range 100 {
		zz := make([]Word, 1+size+1)
		xx := make([]Word, 1+size+1)
		yy := make([]Word, 1+size+1)
		words := words4
		if size > 3 {
			words = words2
		}
		if size > 7 {
			words = nil // random
		}
		for x := range nats(words, size) {
			for y := range nats(words, size) {
				for _, m := range ms {
					for _, a := range as {
						wantZ := make([]Word, size)
						wantC := ref(wantZ, x, y, m, a)

						for _, inplace := range []bool{false, true} {
							name := name
							if inplace {
								name = "in-place " + name
							}
							setSlice(xx, 1, x)
							setSlice(yy, 2, y)
							zz := zz
							if inplace {
								zz = xx
							} else {
								for i := range zz {
									zz[i] = 0x9876
								}
							}
							setSlice(zz, 3, nil)
							c := fn(zz[1:1+size], xx[1:1+size], yy[1:1+size], m, a)
							if !slices.Equal(zz[1:1+size], wantZ) || c != wantC {
								t.Errorf("%s(%#x, %#x, %#x, %#x) = %#x, %#x, want %#x, %#x", name, x, y, m, a, zz[1:1+size], c, wantZ, wantC)
							}
							if !inplace {
								checkSlice(t, name, "x", xx, 1, x)
							}
							checkSlice(t, name, "y", yy, 2, y)
							checkSlice(t, name, "z", zz, 3, nil)
							if t.Failed() {
								t.FailNow()
							}
						}
					}
				}
			}
		}
	}
}

const (
	magic0 = 0x123450
	magic1 = 0x543210
)

// setSlice sets x[1:len(x)-1] to orig, leaving magic values in x[0] and x[len(x)-1]
// so that we can tell if routines accidentally write before or after the data.
func setSlice(x []Word, id Word, orig []Word) {
	x[0] = magic0 + id
	copy(x[1:len(x)-1], orig)
	x[len(x)-1] = magic1 + id
}

// checkSlice checks that the magic values left by setSlices are still there.
// If orig != nil, it also checks that the actual data in x is unmodified since setSlice.
func checkSlice(t *testing.T, name, val string, x []Word, id Word, orig []Word) {
	if x[0] != magic0+id {
		t.Errorf("%s smashed %s[-1]", name, val)
	}
	if x[len(x)-1] != magic1+id {
		t.Errorf("%s smashed %s[len(%s)]", name, val, val)
	}
	if orig != nil && !slices.Equal(x[1:len(x)-1], orig) {
		t.Errorf("%s smashed %s: have %d, want %d", name, val, x[1:len(x)-1], orig)
	}
}

// nats returns a sequence of interesting nats of the given size:
//
//   - all 0
//   - all ^0
//   - all possible combinations of words
//   - ten random values
func nats(words []Word, size int) iter.Seq[[]Word] {
	return func(yield func([]Word) bool) {
		if size == 0 {
			yield(nil)
			return
		}
		w := make([]Word, size)

		// all 0
		for i := range w {
			w[i] = 0
		}
		if !yield(w) {
			return
		}

		// all ^0
		for i := range w {
			w[i] = _M
		}
		if !yield(w) {
			return
		}

		// all possible combinations of words
		var generate func(int) bool
		generate = func(i int) bool {
			if i >= len(w) {
				return yield(w)
			}
			for _, w[i] = range words {
				if !generate(i + 1) {
					return false
				}
			}
			return true
		}
		if !generate(0) {
			return
		}

		// ten random values
		for range 10 {
			for i := range w {
				w[i] = Word(rnd.Uint())
			}
			if !yield(w) {
				return
			}
		}
	}
}

// Always the same seed for reproducible results.
var rnd = rand.New(rand.NewPCG(1, 2))

func rndW() Word {
	return Word(rnd.Uint())
}

func rndV(n int) []Word {
	v := make([]Word, n)
	for i := range v {
		v[i] = rndW()
	}
	return v
}

// Construct a vector comprising the same word, usually '0' or 'maximum uint'
func makeWordVec(e Word, n int) []Word {
	v := make([]Word, n)
	for i := range v {
		v[i] = e
	}
	return v
}

type argVU struct {
	d  []Word // d is a Word slice, the input parameters x and z come from this array.
	l  uint   // l is the length of the input parameters x and z.
	xp uint   // xp is the starting position of the input parameter x, x := d[xp:xp+l].
	zp uint   // zp is the starting position of the input parameter z, z := d[zp:zp+l].
	s  uint   // s is the shift number.
	r  []Word // r is the expected output result z.
	c  Word   // c is the expected return value.
	m  string // message.
}

var arglshVUIn = []Word{1, 2, 4, 8, 16, 32, 64, 0, 0, 0}
var arglshVUr0 = []Word{1, 2, 4, 8, 16, 32, 64}
var arglshVUr1 = []Word{2, 4, 8, 16, 32, 64, 128}
var arglshVUrWm1 = []Word{1 << (_W - 1), 0, 1, 2, 4, 8, 16}

var arglshVU = []argVU{
	// test cases for lshVU
	{[]Word{1, _M, _M, _M, _M, _M, 3 << (_W - 2), 0}, 7, 0, 0, 1, []Word{2, _M - 1, _M, _M, _M, _M, 1<<(_W-1) + 1}, 1, "complete overlap of lshVU"},
	{[]Word{1, _M, _M, _M, _M, _M, 3 << (_W - 2), 0, 0, 0, 0}, 7, 0, 3, 1, []Word{2, _M - 1, _M, _M, _M, _M, 1<<(_W-1) + 1}, 1, "partial overlap by half of lshVU"},
	{[]Word{1, _M, _M, _M, _M, _M, 3 << (_W - 2), 0, 0, 0, 0, 0, 0, 0}, 7, 0, 6, 1, []Word{2, _M - 1, _M, _M, _M, _M, 1<<(_W-1) + 1}, 1, "partial overlap by 1 Word of lshVU"},
	{[]Word{1, _M, _M, _M, _M, _M, 3 << (_W - 2), 0, 0, 0, 0, 0, 0, 0, 0}, 7, 0, 7, 1, []Word{2, _M - 1, _M, _M, _M, _M, 1<<(_W-1) + 1}, 1, "no overlap of lshVU"},
	// additional test cases with shift values of 1 and (_W-1)
	{arglshVUIn, 7, 0, 0, 1, arglshVUr1, 0, "complete overlap of lshVU and shift of 1"},
	{arglshVUIn, 7, 0, 0, _W - 1, arglshVUrWm1, 32, "complete overlap of lshVU and shift of _W - 1"},
	{arglshVUIn, 7, 0, 1, 1, arglshVUr1, 0, "partial overlap by 6 Words of lshVU and shift of 1"},
	{arglshVUIn, 7, 0, 1, _W - 1, arglshVUrWm1, 32, "partial overlap by 6 Words of lshVU and shift of _W - 1"},
	{arglshVUIn, 7, 0, 2, 1, arglshVUr1, 0, "partial overlap by 5 Words of lshVU and shift of 1"},
	{arglshVUIn, 7, 0, 2, _W - 1, arglshVUrWm1, 32, "partial overlap by 5 Words of lshVU abd shift of _W - 1"},
	{arglshVUIn, 7, 0, 3, 1, arglshVUr1, 0, "partial overlap by 4 Words of lshVU and shift of 1"},
	{arglshVUIn, 7, 0, 3, _W - 1, arglshVUrWm1, 32, "partial overlap by 4 Words of lshVU and shift of _W - 1"},
}

var argrshVUIn = []Word{0, 0, 0, 1, 2, 4, 8, 16, 32, 64}
var argrshVUr0 = []Word{1, 2, 4, 8, 16, 32, 64}
var argrshVUr1 = []Word{0, 1, 2, 4, 8, 16, 32}
var argrshVUrWm1 = []Word{4, 8, 16, 32, 64, 128, 0}

var argrshVU = []argVU{
	// test cases for rshVU
	{[]Word{0, 3, _M, _M, _M, _M, _M, 1 << (_W - 1)}, 7, 1, 1, 1, []Word{1<<(_W-1) + 1, _M, _M, _M, _M, _M >> 1, 1 << (_W - 2)}, 1 << (_W - 1), "complete overlap of rshVU"},
	{[]Word{0, 0, 0, 0, 3, _M, _M, _M, _M, _M, 1 << (_W - 1)}, 7, 4, 1, 1, []Word{1<<(_W-1) + 1, _M, _M, _M, _M, _M >> 1, 1 << (_W - 2)}, 1 << (_W - 1), "partial overlap by half of rshVU"},
	{[]Word{0, 0, 0, 0, 0, 0, 0, 3, _M, _M, _M, _M, _M, 1 << (_W - 1)}, 7, 7, 1, 1, []Word{1<<(_W-1) + 1, _M, _M, _M, _M, _M >> 1, 1 << (_W - 2)}, 1 << (_W - 1), "partial overlap by 1 Word of rshVU"},
	{[]Word{0, 0, 0, 0, 0, 0, 0, 0, 3, _M, _M, _M, _M, _M, 1 << (_W - 1)}, 7, 8, 1, 1, []Word{1<<(_W-1) + 1, _M, _M, _M, _M, _M >> 1, 1 << (_W - 2)}, 1 << (_W - 1), "no overlap of rshVU"},
	// additional test cases with shift values of 0, 1 and (_W-1)
	{argrshVUIn, 7, 3, 3, 1, argrshVUr1, 1 << (_W - 1), "complete overlap of rshVU and shift of 1"},
	{argrshVUIn, 7, 3, 3, _W - 1, argrshVUrWm1, 2, "complete overlap of rshVU and shift of _W - 1"},
	{argrshVUIn, 7, 3, 2, 1, argrshVUr1, 1 << (_W - 1), "partial overlap by 6 Words of rshVU and shift of 1"},
	{argrshVUIn, 7, 3, 2, _W - 1, argrshVUrWm1, 2, "partial overlap by 6 Words of rshVU and shift of _W - 1"},
	{argrshVUIn, 7, 3, 1, 1, argrshVUr1, 1 << (_W - 1), "partial overlap by 5 Words of rshVU and shift of 1"},
	{argrshVUIn, 7, 3, 1, _W - 1, argrshVUrWm1, 2, "partial overlap by 5 Words of rshVU and shift of _W - 1"},
	{argrshVUIn, 7, 3, 0, 1, argrshVUr1, 1 << (_W - 1), "partial overlap by 4 Words of rshVU and shift of 1"},
	{argrshVUIn, 7, 3, 0, _W - 1, argrshVUrWm1, 2, "partial overlap by 4 Words of rshVU and shift of _W - 1"},
}

func testShiftFunc(t *testing.T, f func(z, x []Word, s uint) Word, a argVU) {
	// work on copy of a.d to preserve the original data.
	b := make([]Word, len(a.d))
	copy(b, a.d)
	z := b[a.zp : a.zp+a.l]
	x := b[a.xp : a.xp+a.l]
	c := f(z, x, a.s)
	for i, zi := range z {
		if zi != a.r[i] {
			t.Errorf("d := %v, %s (d[%d:%d], d[%d:%d], %d)\n\tgot z[%d] = %#x; want %#x", a.d, a.m, a.zp, a.zp+a.l, a.xp, a.xp+a.l, a.s, i, zi, a.r[i])
			break
		}
	}
	if c != a.c {
		t.Errorf("d := %v, %s (d[%d:%d], d[%d:%d], %d)\n\tgot c = %#x; want %#x", a.d, a.m, a.zp, a.zp+a.l, a.xp, a.xp+a.l, a.s, c, a.c)
	}
}

func TestShiftOverlap(t *testing.T) {
	for _, a := range arglshVU {
		arg := a
		testShiftFunc(t, lshVU, arg)
	}

	for _, a := range argrshVU {
		arg := a
		testShiftFunc(t, rshVU, arg)
	}
}

func TestIssue31084(t *testing.T) {
	stk := getStack()
	defer stk.free()

	// compute 10^n via 5^n << n.
	const n = 165
	p := nat(nil).expNN(stk, nat{5}, nat{n}, nil, false)
	p = p.lsh(p, n)
	got := string(p.utoa(10))
	want := "1" + strings.Repeat("0", n)
	if got != want {
		t.Errorf("lsh(%v, %v)\n\tgot  %s\n\twant %s", p, n, got, want)
	}
}

const issue42838Value = "159309191113245227702888039776771180559110455519261878607388585338616290151305816094308987472018268594098344692611135542392730712890625"

func TestIssue42838(t *testing.T) {
	const s = 192
	z, _, _, _ := nat(nil).scan(strings.NewReader(issue42838Value), 0, false)
	z = z.lsh(z, s)
	got := string(z.utoa(10))
	want := "1" + strings.Repeat("0", s)
	if got != want {
		t.Errorf("lsh(%v, %v)\n\tgot  %s\n\twant %s", z, s, got, want)
	}
}

type funVWW func(z, x []Word, y, r Word) (c Word)
type argVWW struct {
	z, x nat
	y, r Word
	c    Word
}

var prodVWW = []argVWW{
	{},
	{nat{0}, nat{0}, 0, 0, 0},
	{nat{991}, nat{0}, 0, 991, 0},
	{nat{0}, nat{_M}, 0, 0, 0},
	{nat{991}, nat{_M}, 0, 991, 0},
	{nat{0}, nat{0}, _M, 0, 0},
	{nat{991}, nat{0}, _M, 991, 0},
	{nat{1}, nat{1}, 1, 0, 0},
	{nat{992}, nat{1}, 1, 991, 0},
	{nat{22793}, nat{991}, 23, 0, 0},
	{nat{22800}, nat{991}, 23, 7, 0},
	{nat{0, 0, 0, 22793}, nat{0, 0, 0, 991}, 23, 0, 0},
	{nat{7, 0, 0, 22793}, nat{0, 0, 0, 991}, 23, 7, 0},
	{nat{0, 0, 0, 0}, nat{7893475, 7395495, 798547395, 68943}, 0, 0, 0},
	{nat{991, 0, 0, 0}, nat{7893475, 7395495, 798547395, 68943}, 0, 991, 0},
	{nat{0, 0, 0, 0}, nat{0, 0, 0, 0}, 894375984, 0, 0},
	{nat{991, 0, 0, 0}, nat{0, 0, 0, 0}, 894375984, 991, 0},
	{nat{_M << 1 & _M}, nat{_M}, 1 << 1, 0, _M >> (_W - 1)},
	{nat{_M<<1&_M + 1}, nat{_M}, 1 << 1, 1, _M >> (_W - 1)},
	{nat{_M << 7 & _M}, nat{_M}, 1 << 7, 0, _M >> (_W - 7)},
	{nat{_M<<7&_M + 1<<6}, nat{_M}, 1 << 7, 1 << 6, _M >> (_W - 7)},
	{nat{_M << 7 & _M, _M, _M, _M}, nat{_M, _M, _M, _M}, 1 << 7, 0, _M >> (_W - 7)},
	{nat{_M<<7&_M + 1<<6, _M, _M, _M}, nat{_M, _M, _M, _M}, 1 << 7, 1 << 6, _M >> (_W - 7)},
}

func testFunVWW(t *testing.T, msg string, f funVWW, a argVWW) {
	z := make(nat, len(a.z))
	c := f(z, a.x, a.y, a.r)
	for i, zi := range z {
		if zi != a.z[i] {
			t.Errorf("%s%+v\n\tgot z[%d] = %#x; want %#x", msg, a, i, zi, a.z[i])
			break
		}
	}
	if c != a.c {
		t.Errorf("%s%+v\n\tgot c = %#x; want %#x", msg, a, c, a.c)
	}
}

// TODO(gri) mulAddVWW and divWVW are symmetric operations but
// their signature is not symmetric. Try to unify.

type funWVW func(z []Word, xn Word, x []Word, y Word) (r Word)
type argWVW struct {
	z  nat
	xn Word
	x  nat
	y  Word
	r  Word
}

func testFunWVW(t *testing.T, msg string, f funWVW, a argWVW) {
	z := make(nat, len(a.z))
	r := f(z, a.xn, a.x, a.y)
	if !slices.Equal(z, a.z) || r != a.r {
		t.Errorf("%s%+v\nhave %v, %v\nwant %v, %v", msg, a, z, r, a.z, a.r)
	} else {
		t.Logf("%s%+v\ngood %v, %v", msg, a, z, r)
	}
}

func TestFunVWW(t *testing.T) {
	for _, a := range prodVWW {
		arg := a
		testFunVWW(t, "mulAddVWW_g", mulAddVWW_g, arg)
		testFunVWW(t, "mulAddVWW", mulAddVWW, arg)

		if a.y != 0 && a.r < a.y {
			arg := argWVW{a.x, a.c, a.z, a.y, a.r}
			testFunWVW(t, "divWVW", divWVW, arg)
		}
	}
}

var mulWWTests = []struct {
	x, y Word
	q, r Word
}{
	{_M, _M, _M - 1, 1},
	// 32 bit only: {0xc47dfa8c, 50911, 0x98a4, 0x998587f4},
}

func TestMulWW(t *testing.T) {
	for i, test := range mulWWTests {
		q, r := mulWW(test.x, test.y)
		if q != test.q || r != test.r {
			t.Errorf("#%d got (%x, %x) want (%x, %x)", i, q, r, test.q, test.r)
		}
	}
}

var mulAddWWWTests = []struct {
	x, y, c Word
	q, r    Word
}{
	// TODO(agl): These will only work on 64-bit platforms.
	// {15064310297182388543, 0xe7df04d2d35d5d80, 13537600649892366549, 13644450054494335067, 10832252001440893781},
	// {15064310297182388543, 0xdab2f18048baa68d, 13644450054494335067, 12869334219691522700, 14233854684711418382},
	{_M, _M, 0, _M - 1, 1},
	{_M, _M, _M, _M, 0},
}

func TestMulAddWWW(t *testing.T) {
	for i, test := range mulAddWWWTests {
		q, r := mulAddWWW_g(test.x, test.y, test.c)
		if q != test.q || r != test.r {
			t.Errorf("#%d got (%x, %x) want (%x, %x)", i, q, r, test.q, test.r)
		}
	}
}

var divWWTests = []struct {
	x1, x0, y Word
	q, r      Word
}{
	{_M >> 1, 0, _M, _M >> 1, _M >> 1},
	{_M - (1 << (_W - 2)), _M, 3 << (_W - 2), _M, _M - (1 << (_W - 2))},
}

const testsNumber = 1 << 16

func TestDivWW(t *testing.T) {
	i := 0
	for i, test := range divWWTests {
		rec := reciprocalWord(test.y)
		q, r := divWW(test.x1, test.x0, test.y, rec)
		if q != test.q || r != test.r {
			t.Errorf("#%d got (%x, %x) want (%x, %x)", i, q, r, test.q, test.r)
		}
	}
	//random tests
	for ; i < testsNumber; i++ {
		x1 := rndW()
		x0 := rndW()
		y := rndW()
		if x1 >= y {
			continue
		}
		rec := reciprocalWord(y)
		qGot, rGot := divWW(x1, x0, y, rec)
		qWant, rWant := bits.Div(uint(x1), uint(x0), uint(y))
		if uint(qGot) != qWant || uint(rGot) != rWant {
			t.Errorf("#%d got (%x, %x) want (%x, %x)", i, qGot, rGot, qWant, rWant)
		}
	}
}

// benchSizes are the benchmark word sizes.
var benchSizes = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 100, 1000, 10_000, 100_000}

// A benchFunc is a function to be benchmarked.
// It takes one output buffer and two input buffers,
// but it does not have to use any of them.
type benchFunc func(z, x, y []Word)

// bench runs benchmarks of fn for a variety of word sizes.
// It adds the given suffix (for example "/impl=go") to the benchmark names it creates,
// after a "/words=N" parameter. Putting words first makes it easier to run
// all benchmarks with a specific word size
// (go test -run=NONE '-bench=V/words=100$')
// even if different benchmarks have different numbers of other parameters.
func bench(b *testing.B, suffix string, fn benchFunc) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		var z, x, y []Word
		b.Run(fmt.Sprintf("words=%d%s", n, suffix), func(b *testing.B) {
			if z == nil {
				z = make([]Word, n)
				x = rndV(n)
				y = rndV(n)
			}
			b.SetBytes(int64(n * _S))
			for b.Loop() {
				fn(z, x, y)
			}
		})
	}
}

// Benchmark basic I/O and arithmetic processing speed,
// to help estimate the upper bounds on other operations.

func BenchmarkCopyVV(b *testing.B) { bench(b, "", benchVV(copyVV)) }

func copyVV(z, x, y []Word) Word {
	copy(z, x)
	return 0
}

// Note: This benchmark consistently runs faster (even up to 2X faster on MB/s)
// with words=10 and words=100 than larger amounts like words=1000 or words=10000.
// The reason appears to that if you run 100-word addition loops repeatedly,
// they are independent calculations, and the processor speculates/pipelines/whatever
// to such a deep level that it can overlap the repeated loops.
// In contrast, if you run 1000-word or 10000-word loops repeatedly,
// the dependency chains are so long that the processor cannot overlap them.
// If we change arithVV to take the starting value of s and pass in the result
// from the previous arithVV, then even the 10-word or 100-loops become
// a single long dependency chain and the 2X disappears. But since we are
// using BenchmarkArithVV for a given word size to estimate the upper bound
// of, say, BenchmarkAddVV for that same word size, we actually want the
// dependency chain-length variation in BenchmarkArithVV too.
// It's just mysterious to see until you understand what is causing it.

func BenchmarkArithVV(b *testing.B) { bench(b, "", benchVV(arithVV)) }

func arithVV(z, x, y []Word) Word {
	var a, b, c, d, e, f, g, h, i, j Word
	if len(z) >= 8 {
		a, b, c, d, e, f, g, h, i, j = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
	}
	if len(z) < 10 {
		// We don't really care about the speed here, but
		// do something so that the small word counts aren't all the same.
		s := Word(0)
		for _, zi := range z {
			s += zi
		}
		return s
	}
	s := Word(0)
	for range len(z) / 10 {
		s += a
		s += b
		s += c
		s += d
		s += e
		s += f
		s += g
		s += h
		s += i
		s += j
	}
	return s
}

func BenchmarkAddVV(b *testing.B) {
	bench(b, "/impl=asm", benchVV(addVV))
	bench(b, "/impl=go", benchVV(addVV_g))
}

func BenchmarkSubVV(b *testing.B) {
	bench(b, "/impl=asm", benchVV(subVV))
	bench(b, "/impl=go", benchVV(subVV_g))
}

func benchVV(fn func(z, x, y []Word) Word) benchFunc {
	return func(z, x, y []Word) { fn(z, x, y) }
}

func BenchmarkAddVW(b *testing.B) {
	bench(b, "/data=random", benchVW(addVW, 123))
	bench(b, "/data=carry", benchCarryVW(addVW, ^Word(0), 1))
	bench(b, "/data=shortcut", benchShortVW(addVW, 123))
}

func BenchmarkSubVW(b *testing.B) {
	bench(b, "/data=random", benchVW(subVW, 123))
	bench(b, "/data=carry", benchCarryVW(subVW, 0, 1))
	bench(b, "/data=shortcut", benchShortVW(subVW, 123))
}

func benchVW(fn func(z, x []Word, w Word) Word, w Word) benchFunc {
	return func(z, x, y []Word) { fn(z, x, w) }
}

func benchCarryVW(fn func(z, x []Word, w Word) Word, xi, w Word) benchFunc {
	return func(z, x, y []Word) {
		// Fill x with xi the first time we are called with a given x.
		// Otherwise x is random, so checking the first two elements is good enough.
		// Assume this is the warmup, so we don't need to worry about it taking longer.
		if x[0] != w || len(x) >= 2 && x[1] != w {
			for i := range x {
				x[i] = xi
			}
		}
		fn(z, x, w)
	}
}

func benchShortVW(fn func(z, x []Word, w Word) Word, w Word) benchFunc {
	// Note: calling fn with x not z, to benchmark in-place overwriting.
	return func(z, x, y []Word) { fn(x, x, w) }
}

func BenchmarkLshVU(b *testing.B) {
	bench(b, "/impl=asm", benchVU(lshVU, 3))
	bench(b, "/impl=go", benchVU(lshVU_g, 3))
}

func BenchmarkRshVU(b *testing.B) {
	bench(b, "/impl=asm", benchVU(rshVU, 3))
	bench(b, "/impl=go", benchVU(rshVU_g, 3))
}

func benchVU(fn func(z, x []Word, s uint) Word, s uint) benchFunc {
	return func(z, x, y []Word) { fn(z, x, s) }
}

func BenchmarkMulAddVWW(b *testing.B) {
	bench(b, "/impl=asm", benchVWW(mulAddVWW, 42, 100))
	bench(b, "/impl=go", benchVWW(mulAddVWW_g, 42, 100))
}

func benchVWW(fn func(z, x []Word, w1, w2 Word) Word, w1, w2 Word) benchFunc {
	return func(z, x, y []Word) { fn(z, x, w1, w2) }
}

func BenchmarkAddMulVVWW(b *testing.B) {
	bench(b, "/impl=asm", benchVVWW(addMulVVWW, 42, 100))
	bench(b, "/impl=go", benchVVWW(addMulVVWW_g, 42, 100))
}

func benchVVWW(fn func(z, x, y []Word, w1, w2 Word) Word, w1, w2 Word) benchFunc {
	return func(z, x, y []Word) { fn(z, x, y, w1, w2) }
}

func BenchmarkDivWVW(b *testing.B) {
	bench(b, "", benchWVW(divWVW, 100, 200))
}

func benchWVW(fn func(z []Word, w1 Word, x []Word, w2 Word) Word, w1, w2 Word) benchFunc {
	return func(z, x, y []Word) { fn(z, w1, x, w2) }
}
