// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constant_test

import (
	"fmt"
	"go/constant"
	"go/token"
	"math"
	"sort"
)

func Example_complexNumbers() {
	// Create the complex number 2.3 + 5i.
	ar := constant.MakeFloat64(2.3)
	ai := constant.MakeImag(constant.MakeInt64(5))
	a := constant.BinaryOp(ar, token.ADD, ai)

	// Compute (2.3 + 5i) * 11.
	b := constant.MakeUint64(11)
	c := constant.BinaryOp(a, token.MUL, b)

	// Convert c into a complex128.
	Ar, exact := constant.Float64Val(constant.Real(c))
	if !exact {
		fmt.Printf("Could not represent real part %s exactly as float64\n", constant.Real(c))
	}
	Ai, exact := constant.Float64Val(constant.Imag(c))
	if !exact {
		fmt.Printf("Could not represent imaginary part %s as exactly as float64\n", constant.Imag(c))
	}
	C := complex(Ar, Ai)

	fmt.Println("literal", 25.3+55i)
	fmt.Println("go/constant", c)
	fmt.Println("complex128", C)

	// Output:
	//
	// Could not represent real part 25.3 exactly as float64
	// literal (25.3+55i)
	// go/constant (25.3 + 55i)
	// complex128 (25.299999999999997+55i)
}

func ExampleBinaryOp() {
	// 11 / 0.5
	a := constant.MakeUint64(11)
	b := constant.MakeFloat64(0.5)
	c := constant.BinaryOp(a, token.QUO, b)
	fmt.Println(c)

	// Output: 22
}

func ExampleUnaryOp() {
	vs := []constant.Value{
		constant.MakeBool(true),
		constant.MakeFloat64(2.7),
		constant.MakeUint64(42),
	}

	for i, v := range vs {
		switch v.Kind() {
		case constant.Bool:
			vs[i] = constant.UnaryOp(token.NOT, v, 0)

		case constant.Float:
			vs[i] = constant.UnaryOp(token.SUB, v, 0)

		case constant.Int:
			// Use 16-bit precision.
			// This would be equivalent to ^uint16(v).
			vs[i] = constant.UnaryOp(token.XOR, v, 16)
		}
	}

	for _, v := range vs {
		fmt.Println(v)
	}

	// Output:
	//
	// false
	// -2.7
	// 65493
}

func ExampleCompare() {
	vs := []constant.Value{
		constant.MakeString("Z"),
		constant.MakeString("bacon"),
		constant.MakeString("go"),
		constant.MakeString("Frame"),
		constant.MakeString("defer"),
		constant.MakeFromLiteral(`"a"`, token.STRING, 0),
	}

	sort.Slice(vs, func(i, j int) bool {
		// Equivalent to vs[i] <= vs[j].
		return constant.Compare(vs[i], token.LEQ, vs[j])
	})

	for _, v := range vs {
		fmt.Println(constant.StringVal(v))
	}

	// Output:
	//
	// Frame
	// Z
	// a
	// bacon
	// defer
	// go
}

func ExampleSign() {
	zero := constant.MakeInt64(0)
	one := constant.MakeInt64(1)
	negOne := constant.MakeInt64(-1)

	mkComplex := func(a, b constant.Value) constant.Value {
		b = constant.MakeImag(b)
		return constant.BinaryOp(a, token.ADD, b)
	}

	vs := []constant.Value{
		negOne,
		mkComplex(zero, negOne),
		mkComplex(one, negOne),
		mkComplex(negOne, one),
		mkComplex(negOne, negOne),
		zero,
		mkComplex(zero, zero),
		one,
		mkComplex(zero, one),
		mkComplex(one, one),
	}

	for _, v := range vs {
		fmt.Printf("% d %s\n", constant.Sign(v), v)
	}

	// Output:
	//
	// -1 -1
	// -1 (0 + -1i)
	// -1 (1 + -1i)
	// -1 (-1 + 1i)
	// -1 (-1 + -1i)
	//  0 0
	//  0 (0 + 0i)
	//  1 1
	//  1 (0 + 1i)
	//  1 (1 + 1i)
}

func ExampleVal() {
	maxint := constant.MakeInt64(math.MaxInt64)
	fmt.Printf("%v\n", constant.Val(maxint))

	e := constant.MakeFloat64(math.E)
	fmt.Printf("%v\n", constant.Val(e))

	b := constant.MakeBool(true)
	fmt.Printf("%v\n", constant.Val(b))

	b = constant.Make(false)
	fmt.Printf("%v\n", constant.Val(b))

	// Output:
	//
	// 9223372036854775807
	// 6121026514868073/2251799813685248
	// true
	// false
}
