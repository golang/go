// runoutput

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/big"
	"unsafe"
)

var one = big.NewInt(1)

type _type struct {
	name   string
	bits   uint
	signed bool
}

// testvalues returns a list of all test values for this type.
func (t *_type) testvalues() []*big.Int {
	var a []*big.Int

	a = append(a, big.NewInt(0))
	a = append(a, big.NewInt(1))
	a = append(a, big.NewInt(2))
	if t.signed {
		a = append(a, big.NewInt(-1))
		a = append(a, big.NewInt(-2))
		r := big.NewInt(1)
		a = append(a, r.Lsh(r, t.bits-1).Sub(r, big.NewInt(1)))
		r = big.NewInt(1)
		a = append(a, r.Lsh(r, t.bits-1).Sub(r, big.NewInt(2)))
		r = big.NewInt(1)
		a = append(a, r.Lsh(r, t.bits-1).Neg(r))
		r = big.NewInt(1)
		a = append(a, r.Lsh(r, t.bits-1).Neg(r).Add(r, big.NewInt(1)))
	} else {
		r := big.NewInt(1)
		a = append(a, r.Lsh(r, t.bits).Sub(r, big.NewInt(1)))
		r = big.NewInt(1)
		a = append(a, r.Lsh(r, t.bits).Sub(r, big.NewInt(2)))
	}
	return a
}

// trunc truncates a value to the range of the given type.
func (t *_type) trunc(x *big.Int) *big.Int {
	r := new(big.Int)
	m := new(big.Int)
	m.Lsh(one, t.bits)
	m.Sub(m, one)
	r.And(x, m)
	if t.signed && r.Bit(int(t.bits)-1) == 1 {
		m.Neg(one)
		m.Lsh(m, t.bits)
		r.Or(r, m)
	}
	return r
}

var types = []_type{
	_type{"byte", 8, false},
	_type{"int8", 8, true},
	_type{"uint8", 8, false},
	_type{"rune", 32, true},
	_type{"int16", 16, true},
	_type{"uint16", 16, false},
	_type{"int32", 32, true},
	_type{"uint32", 32, false},
	_type{"int64", 64, true},
	_type{"uint64", 64, false},
	_type{"int", 8 * uint(unsafe.Sizeof(int(0))), true},
	_type{"uint", 8 * uint(unsafe.Sizeof(uint(0))), false},
	_type{"uintptr", 8 * uint(unsafe.Sizeof((*byte)(nil))), false},
}

type binop struct {
	name string
	eval func(x, y *big.Int) *big.Int
}

var binops = []binop{
	binop{"+", func(x, y *big.Int) *big.Int { return new(big.Int).Add(x, y) }},
	binop{"-", func(x, y *big.Int) *big.Int { return new(big.Int).Sub(x, y) }},
	binop{"*", func(x, y *big.Int) *big.Int { return new(big.Int).Mul(x, y) }},
	binop{"/", func(x, y *big.Int) *big.Int { return new(big.Int).Quo(x, y) }},
	binop{"%", func(x, y *big.Int) *big.Int { return new(big.Int).Rem(x, y) }},
	binop{"&", func(x, y *big.Int) *big.Int { return new(big.Int).And(x, y) }},
	binop{"|", func(x, y *big.Int) *big.Int { return new(big.Int).Or(x, y) }},
	binop{"^", func(x, y *big.Int) *big.Int { return new(big.Int).Xor(x, y) }},
	binop{"&^", func(x, y *big.Int) *big.Int { return new(big.Int).AndNot(x, y) }},
}

type unop struct {
	name string
	eval func(x *big.Int) *big.Int
}

var unops = []unop{
	unop{"+", func(x *big.Int) *big.Int { return new(big.Int).Set(x) }},
	unop{"-", func(x *big.Int) *big.Int { return new(big.Int).Neg(x) }},
	unop{"^", func(x *big.Int) *big.Int { return new(big.Int).Not(x) }},
}

type shiftop struct {
	name string
	eval func(x *big.Int, i uint) *big.Int
}

var shiftops = []shiftop{
	shiftop{"<<", func(x *big.Int, i uint) *big.Int { return new(big.Int).Lsh(x, i) }},
	shiftop{">>", func(x *big.Int, i uint) *big.Int { return new(big.Int).Rsh(x, i) }},
}

// valname returns the name of n as can be used as part of a variable name.
func valname(n *big.Int) string {
	s := fmt.Sprintf("%d", n)
	if s[0] == '-' {
		s = "neg" + s[1:]
	}
	return s
}

func main() {
	fmt.Println("package main")

	// We make variables to hold all the different values we'd like to use.
	// We use global variables to prevent any constant folding.
	for _, t := range types {
		for _, n := range t.testvalues() {
			fmt.Printf("var %s_%s %s = %d\n", t.name, valname(n), t.name, n)
		}
	}

	fmt.Println("func main() {")

	for _, t := range types {
		// test binary ops
		for _, op := range binops {
			for _, x := range t.testvalues() {
				for _, y := range t.testvalues() {
					if (op.name == "/" || op.name == "%") && y.Sign() == 0 {
						continue
					}
					r := t.trunc(op.eval(x, y))
					eqn := fmt.Sprintf("%s_%s %s %s_%s != %d", t.name, valname(x), op.name, t.name, valname(y), r)
					fmt.Printf("\tif %s { println(\"bad: %s\") }\n", eqn, eqn)
				}
			}
		}
		// test unary ops
		for _, op := range unops {
			for _, x := range t.testvalues() {
				r := t.trunc(op.eval(x))
				eqn := fmt.Sprintf("%s %s_%s != %d", op.name, t.name, valname(x), r)
				fmt.Printf("\tif %s { println(\"bad: %s\") }\n", eqn, eqn)
			}
		}
		// test shifts
		for _, op := range shiftops {
			for _, x := range t.testvalues() {

				for _, i := range []uint{0, 1, t.bits - 2, t.bits - 1, t.bits, t.bits + 1} {
					r := t.trunc(op.eval(x, i))
					eqn := fmt.Sprintf("%s_%s %s %d != %d", t.name, valname(x), op.name, i, r)
					fmt.Printf("\tif %s { println(\"bad: %s\") }\n", eqn, eqn)
				}
			}
		}
	}

	fmt.Println("}")
}
