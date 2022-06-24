// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big_test

import (
	cryptorand "crypto/rand"
	"math/big"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

func equal(z, x *big.Int) bool {
	return z.Cmp(x) == 0
}

type bigInt struct {
	*big.Int
}

func generatePositiveInt(rand *rand.Rand, size int) *big.Int {
	n := big.NewInt(1)
	n.Lsh(n, uint(rand.Intn(size*8)))
	n.Rand(rand, n)
	return n
}

func (bigInt) Generate(rand *rand.Rand, size int) reflect.Value {
	n := generatePositiveInt(rand, size)
	if rand.Intn(4) == 0 {
		n.Neg(n)
	}
	return reflect.ValueOf(bigInt{n})
}

type notZeroInt struct {
	*big.Int
}

func (notZeroInt) Generate(rand *rand.Rand, size int) reflect.Value {
	n := generatePositiveInt(rand, size)
	if rand.Intn(4) == 0 {
		n.Neg(n)
	}
	if n.Sign() == 0 {
		n.SetInt64(1)
	}
	return reflect.ValueOf(notZeroInt{n})
}

type positiveInt struct {
	*big.Int
}

func (positiveInt) Generate(rand *rand.Rand, size int) reflect.Value {
	n := generatePositiveInt(rand, size)
	return reflect.ValueOf(positiveInt{n})
}

type prime struct {
	*big.Int
}

func (prime) Generate(r *rand.Rand, size int) reflect.Value {
	n, err := cryptorand.Prime(r, r.Intn(size*8-2)+2)
	if err != nil {
		panic(err)
	}
	return reflect.ValueOf(prime{n})
}

type zeroOrOne struct {
	uint
}

func (zeroOrOne) Generate(rand *rand.Rand, size int) reflect.Value {
	return reflect.ValueOf(zeroOrOne{uint(rand.Intn(2))})
}

type smallUint struct {
	uint
}

func (smallUint) Generate(rand *rand.Rand, size int) reflect.Value {
	return reflect.ValueOf(smallUint{uint(rand.Intn(1024))})
}

// checkAliasingOneArg checks if f returns a correct result when v and x alias.
//
// f is a function that takes x as an argument, doesn't modify it, sets v to the
// result, and returns v. It is the function signature of unbound methods like
//
//	func (v *big.Int) m(x *big.Int) *big.Int
//
// v and x are two random Int values. v is randomized even if it will be
// overwritten to test for improper buffer reuse.
func checkAliasingOneArg(t *testing.T, f func(v, x *big.Int) *big.Int, v, x *big.Int) bool {
	x1, v1 := new(big.Int).Set(x), new(big.Int).Set(x)

	// Calculate a reference f(x) without aliasing.
	if out := f(v, x); out != v {
		return false
	}

	// Test aliasing the argument and the receiver.
	if out := f(v1, v1); out != v1 || !equal(v1, v) {
		t.Logf("f(v, x) != f(x, x)")
		return false
	}

	// Ensure the arguments was not modified.
	return equal(x, x1)
}

// checkAliasingTwoArgs checks if f returns a correct result when any
// combination of v, x and y alias.
//
// f is a function that takes x and y as arguments, doesn't modify them, sets v
// to the result, and returns v. It is the function signature of unbound methods
// like
//
//	func (v *big.Int) m(x, y *big.Int) *big.Int
//
// v, x and y are random Int values. v is randomized even if it will be
// overwritten to test for improper buffer reuse.
func checkAliasingTwoArgs(t *testing.T, f func(v, x, y *big.Int) *big.Int, v, x, y *big.Int) bool {
	x1, y1, v1 := new(big.Int).Set(x), new(big.Int).Set(y), new(big.Int).Set(v)

	// Calculate a reference f(x, y) without aliasing.
	if out := f(v, x, y); out == nil {
		// Certain functions like ModInverse return nil for certain inputs.
		// Check that receiver and arguments were unchanged and move on.
		return equal(x, x1) && equal(y, y1) && equal(v, v1)
	} else if out != v {
		return false
	}

	// Test aliasing the first argument and the receiver.
	v1.Set(x)
	if out := f(v1, v1, y); out != v1 || !equal(v1, v) {
		t.Logf("f(v, x, y) != f(x, x, y)")
		return false
	}
	// Test aliasing the second argument and the receiver.
	v1.Set(y)
	if out := f(v1, x, v1); out != v1 || !equal(v1, v) {
		t.Logf("f(v, x, y) != f(y, x, y)")
		return false
	}

	// Calculate a reference f(y, y) without aliasing.
	// We use y because it's the one that commonly has restrictions
	// like being prime or non-zero.
	v1.Set(v)
	y2 := new(big.Int).Set(y)
	if out := f(v, y, y2); out == nil {
		return equal(y, y1) && equal(y2, y1) && equal(v, v1)
	} else if out != v {
		return false
	}

	// Test aliasing the two arguments.
	if out := f(v1, y, y); out != v1 || !equal(v1, v) {
		t.Logf("f(v, y1, y2) != f(v, y, y)")
		return false
	}
	// Test aliasing the two arguments and the receiver.
	v1.Set(y)
	if out := f(v1, v1, v1); out != v1 || !equal(v1, v) {
		t.Logf("f(v, y1, y2) != f(y, y, y)")
		return false
	}

	// Ensure the arguments were not modified.
	return equal(x, x1) && equal(y, y1)
}

func TestAliasing(t *testing.T) {
	for name, f := range map[string]interface{}{
		"Abs": func(v, x bigInt) bool {
			return checkAliasingOneArg(t, (*big.Int).Abs, v.Int, x.Int)
		},
		"Add": func(v, x, y bigInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Add, v.Int, x.Int, y.Int)
		},
		"And": func(v, x, y bigInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).And, v.Int, x.Int, y.Int)
		},
		"AndNot": func(v, x, y bigInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).AndNot, v.Int, x.Int, y.Int)
		},
		"Div": func(v, x bigInt, y notZeroInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Div, v.Int, x.Int, y.Int)
		},
		"Exp-XY": func(v, x, y bigInt, z notZeroInt) bool {
			return checkAliasingTwoArgs(t, func(v, x, y *big.Int) *big.Int {
				return v.Exp(x, y, z.Int)
			}, v.Int, x.Int, y.Int)
		},
		"Exp-XZ": func(v, x, y bigInt, z notZeroInt) bool {
			return checkAliasingTwoArgs(t, func(v, x, z *big.Int) *big.Int {
				return v.Exp(x, y.Int, z)
			}, v.Int, x.Int, z.Int)
		},
		"Exp-YZ": func(v, x, y bigInt, z notZeroInt) bool {
			return checkAliasingTwoArgs(t, func(v, y, z *big.Int) *big.Int {
				return v.Exp(x.Int, y, z)
			}, v.Int, y.Int, z.Int)
		},
		"GCD": func(v, x, y bigInt) bool {
			return checkAliasingTwoArgs(t, func(v, x, y *big.Int) *big.Int {
				return v.GCD(nil, nil, x, y)
			}, v.Int, x.Int, y.Int)
		},
		"GCD-X": func(v, x, y bigInt) bool {
			a, b := new(big.Int), new(big.Int)
			return checkAliasingTwoArgs(t, func(v, x, y *big.Int) *big.Int {
				a.GCD(v, b, x, y)
				return v
			}, v.Int, x.Int, y.Int)
		},
		"GCD-Y": func(v, x, y bigInt) bool {
			a, b := new(big.Int), new(big.Int)
			return checkAliasingTwoArgs(t, func(v, x, y *big.Int) *big.Int {
				a.GCD(b, v, x, y)
				return v
			}, v.Int, x.Int, y.Int)
		},
		"Lsh": func(v, x bigInt, n smallUint) bool {
			return checkAliasingOneArg(t, func(v, x *big.Int) *big.Int {
				return v.Lsh(x, n.uint)
			}, v.Int, x.Int)
		},
		"Mod": func(v, x bigInt, y notZeroInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Mod, v.Int, x.Int, y.Int)
		},
		"ModInverse": func(v, x bigInt, y notZeroInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).ModInverse, v.Int, x.Int, y.Int)
		},
		"ModSqrt": func(v, x bigInt, p prime) bool {
			return checkAliasingTwoArgs(t, (*big.Int).ModSqrt, v.Int, x.Int, p.Int)
		},
		"Mul": func(v, x, y bigInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Mul, v.Int, x.Int, y.Int)
		},
		"Neg": func(v, x bigInt) bool {
			return checkAliasingOneArg(t, (*big.Int).Neg, v.Int, x.Int)
		},
		"Not": func(v, x bigInt) bool {
			return checkAliasingOneArg(t, (*big.Int).Not, v.Int, x.Int)
		},
		"Or": func(v, x, y bigInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Or, v.Int, x.Int, y.Int)
		},
		"Quo": func(v, x bigInt, y notZeroInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Quo, v.Int, x.Int, y.Int)
		},
		"Rand": func(v, x bigInt, seed int64) bool {
			return checkAliasingOneArg(t, func(v, x *big.Int) *big.Int {
				rnd := rand.New(rand.NewSource(seed))
				return v.Rand(rnd, x)
			}, v.Int, x.Int)
		},
		"Rem": func(v, x bigInt, y notZeroInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Rem, v.Int, x.Int, y.Int)
		},
		"Rsh": func(v, x bigInt, n smallUint) bool {
			return checkAliasingOneArg(t, func(v, x *big.Int) *big.Int {
				return v.Rsh(x, n.uint)
			}, v.Int, x.Int)
		},
		"Set": func(v, x bigInt) bool {
			return checkAliasingOneArg(t, (*big.Int).Set, v.Int, x.Int)
		},
		"SetBit": func(v, x bigInt, i smallUint, b zeroOrOne) bool {
			return checkAliasingOneArg(t, func(v, x *big.Int) *big.Int {
				return v.SetBit(x, int(i.uint), b.uint)
			}, v.Int, x.Int)
		},
		"Sqrt": func(v bigInt, x positiveInt) bool {
			return checkAliasingOneArg(t, (*big.Int).Sqrt, v.Int, x.Int)
		},
		"Sub": func(v, x, y bigInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Sub, v.Int, x.Int, y.Int)
		},
		"Xor": func(v, x, y bigInt) bool {
			return checkAliasingTwoArgs(t, (*big.Int).Xor, v.Int, x.Int, y.Int)
		},
	} {
		t.Run(name, func(t *testing.T) {
			scale := 1.0
			switch name {
			case "ModInverse", "GCD-Y", "GCD-X":
				scale /= 5
			case "Rand":
				scale /= 10
			case "Exp-XZ", "Exp-XY", "Exp-YZ":
				scale /= 50
			case "ModSqrt":
				scale /= 500
			}
			if err := quick.Check(f, &quick.Config{
				MaxCountScale: scale,
			}); err != nil {
				t.Error(err)
			}
		})
	}
}
