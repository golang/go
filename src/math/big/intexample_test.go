// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big_test

import (
	"fmt"
	. "math/big"
	"math/rand"
	"time"
)

func ExampleNewInt() {
	fmt.Println(NewInt(1))
	// Output:
	// 1
}

func ExampleInt_Abs() {
	z := NewInt(-1)
	fmt.Println(z.Abs(z))
	// Output:
	// 1
}

func ExampleInt_Add() {
	fmt.Println(NewInt(-1).Add(NewInt(1), NewInt(1)))
	// Output:
	// 2
}

func ExampleInt_And() {
	fmt.Println(NewInt(1).And(NewInt(1), NewInt(0)))
	// Output:
	// 0
}

func ExampleInt_AndNot() {
	fmt.Println(NewInt(1).AndNot(NewInt(1), NewInt(0)))
	// Output:
	// 1
}

func ExampleInt_Append() {
	fmt.Println(string(NewInt(10).Append(nil, 10)))
	// Output:
	// 10
}

func ExampleInt_Binomial() {
	fmt.Println(NewInt(0).Binomial(2, 2))
	// Output:
	// 1
}

func ExampleInt_Bit() {
	z := NewInt(1)
	fmt.Println(z.Bit(z.BitLen() - 1))
	// Output:
	// 1
}

func ExampleInt_BitLen() {
	fmt.Println(NewInt(1).BitLen())
	// Output:
	// 1
}

func ExampleInt_Bits() {
	i := NewInt(1)
	b := i.Bits()
	fmt.Println(b)
	// Output:
	// [1]
}

func ExampleInt_Bytes() {
	fmt.Println(NewInt(10).Bytes())
	// Output:
	// [10]
}

func ExampleInt_Cmp() {
	fmt.Println(NewInt(0).Cmp(NewInt(1)))
	// Output:
	// -1
}

func ExampleInt_CmpAbs() {
	fmt.Println(NewInt(0).CmpAbs(NewInt(1)))
	// Output:
	// -1
}

func ExampleInt_Div() {
	fmt.Println(NewInt(2).Div(NewInt(2), NewInt(1)))
	// Output:
	// 2
}

func ExampleInt_DivMod() {
	fmt.Println(NewInt(3).DivMod(NewInt(3), NewInt(2), NewInt(1)))
	// Output:
	// 1 1
}

func ExampleInt_Exp() {
	fmt.Println(NewInt(2).Exp(NewInt(1), NewInt(1), NewInt(1)))
	// Output:
	// 0
}

func ExampleInt_FillBytes() {
	fmt.Println(NewInt(0).FillBytes([]byte{'1'}))
	// Output:
	// [0]
}

func ExampleInt_Float64() {
	fmt.Println(NewInt(0).Float64())
	// Output:
	// 0 Exact
}

func ExampleInt_GCD() {
	fmt.Println(NewInt(0).GCD(nil, nil, NewInt(5), NewInt(10)))
	// Output:
	// 5
}

func ExampleInt_Int64() {
	fmt.Println(NewInt(19).Int64())
	// Output:
	// 19
}

func ExampleInt_IsInt64() {
	fmt.Println(NewInt(19).IsInt64())
	// Output:
	// true
}

func ExampleInt_IsUint64() {
	fmt.Println(NewInt(-19).IsUint64())
	// Output:
	// false
}

func ExampleInt_Lsh() {
	z := NewInt(1)
	fmt.Println(z.Lsh(z, 1))
	// Output:
	// 2
}

func ExampleInt_Mod() {
	fmt.Println(NewInt(2).Mod(NewInt(2), NewInt(2)))
	// Output:
	// 0
}

func ExampleInt_ModInverse() {
	fmt.Println(NewInt(3).ModInverse(NewInt(3), NewInt(5)))
	// Output:
	// 2
}

func ExampleInt_ModSqrt() {
	fmt.Println(NewInt(9).ModInverse(NewInt(14), NewInt(5)))
	// Output:
	// 4
}

func ExampleInt_Mul() {
	fmt.Println(NewInt(9).Mul(NewInt(3), NewInt(3)))
	// Output:
	// 9
}

func ExampleInt_MulRange() {
	fmt.Println(NewInt(9).MulRange(1, 3))
	// Output:
	// 6
}

func ExampleInt_Neg() {
	z := NewInt(9)
	fmt.Println(z.Neg(z))
	// Output:
	// -9
}

func ExampleInt_Not() {
	z := NewInt(0)
	fmt.Println(z.Not(z))
	// Output:
	// -1
}

func ExampleInt_Or() {
	z := NewInt(1)
	fmt.Println(z.Or(z, z))
	// Output:
	// 1
}

func ExampleInt_ProbablyPrime() {
	z := NewInt(7)
	fmt.Println(z.ProbablyPrime(7))
	// Output:
	// true
}

func ExampleInt_Quo() {
	z := NewInt(1)
	fmt.Println(z.Quo(z, z))
	// Output:
	// 1
}

func ExampleInt_QuoRem() {
	z := NewInt(2)
	y := NewInt(1)
	z.QuoRem(z, z, y)
	fmt.Println(z, y)
	// Output:
	// 1 0
}

func ExampleInt_Rem() {
	z := NewInt(1)
	fmt.Println(z.Rem(z, z))
	// Output:
	// 0
}

func ExampleInt_Rand() {
	z := NewInt(1)
	z.Rand(rand.New(rand.NewSource(time.Now().UnixNano())), z)
	z.Add(z, NewInt(1))
	fmt.Println(z.Div(z, z))
	// Output:
	// 1
}

func ExampleInt_Rsh() {
	z := NewInt(1)
	fmt.Println(z.Rsh(z, 1))
	// Output:
	// 0
}

func ExampleInt_Set() {
	z := NewInt(1)
	fmt.Println(z.Set(NewInt(0)))
	// Output:
	// 0
}

func ExampleInt_SetBit() {
	z := NewInt(0)
	fmt.Println(z.SetBit(z, 1, 0))
	fmt.Println(z.SetBit(z, 1, 1))
	// Output:
	// 0
	// 2
}

func ExampleInt_SetBits() {
	z := NewInt(1)
	fmt.Println(z.SetBits([]Word{0}))
	// Output:
	// 0
}

func ExampleInt_SetBytes() {
	z := NewInt(1)
	fmt.Println(z.SetBytes([]byte{byte(0)}))
	// Output:
	// 0
}

func ExampleInt_SetInt64() {
	z := NewInt(1)
	fmt.Println(z.SetInt64(0))
	// Output:
	// 0
}

func ExampleInt_SetUint64() {
	z := NewInt(1)
	fmt.Println(z.SetUint64(0))
	// Output:
	// 0
}

func ExampleInt_Sign() {
	z := NewInt(1)
	fmt.Println(z.Sign())
	// Output:
	// 1
}

func ExampleInt_Sqrt() {
	z := NewInt(0)
	fmt.Println(z.Sqrt(NewInt(4)))
	// Output:
	// 2
}

func ExampleInt_Sub() {
	z := NewInt(1)
	fmt.Println(z.Sub(z, z))
	// Output:
	// 0
}

func ExampleInt_Text() {
	z := NewInt(0)
	fmt.Println(z.Text(10))
	// Output:
	// 0
}

func ExampleInt_TrailingZeroBits() {
	z := NewInt(0)
	fmt.Println(z.TrailingZeroBits())
	z.SetInt64(2)
	fmt.Println(z.TrailingZeroBits())
	z.SetInt64(-2)
	fmt.Println(z.TrailingZeroBits())
	// Output:
	// 0
	// 1
	// 1
}

func ExampleInt_Uint64() {
	z := NewInt(0)
	fmt.Println(z.Uint64())
	// Output:
	// 0
}

func ExampleInt_Xor() {
	z := NewInt(0)
	fmt.Println(z.Xor(z, z))
	// Output:
	// 0
}
