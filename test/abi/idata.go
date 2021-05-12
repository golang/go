// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Excerpted from go/constant/value.go to capture a bug from there.

package main

import (
	"fmt"
	"math"
	"math/big"
)

type (
	unknownVal struct{}
	intVal     struct{ val *big.Int }   // Int values not representable as an int64
	ratVal     struct{ val *big.Rat }   // Float values representable as a fraction
	floatVal   struct{ val *big.Float } // Float values not representable as a fraction
	complexVal struct{ re, im Value }
)

const prec = 512

func (unknownVal) String() string { return "unknown" }

func (x intVal) String() string   { return x.val.String() }
func (x ratVal) String() string   { return rtof(x).String() }

func (x floatVal) String() string {
	f := x.val

	// Use exact fmt formatting if in float64 range (common case):
	// proceed if f doesn't underflow to 0 or overflow to inf.
	if x, _ := f.Float64(); f.Sign() == 0 == (x == 0) && !math.IsInf(x, 0) {
		return fmt.Sprintf("%.6g", x)
	}

	return "OOPS"
}

func (x complexVal) String() string { return fmt.Sprintf("(%s + %si)", x.re, x.im) }

func newFloat() *big.Float { return new(big.Float).SetPrec(prec) }

//go:noinline
//go:registerparams
func itor(x intVal) ratVal       { return ratVal{nil} }

//go:noinline
//go:registerparams
func itof(x intVal) floatVal     { return floatVal{nil} }
func rtof(x ratVal) floatVal     { return floatVal{newFloat().SetRat(x.val)} }

type Value interface {
	String() string
}

//go:noinline
//go:registerparams
func ToFloat(x Value) Value {
	switch x := x.(type) {
	case intVal:
		if smallInt(x.val) {
			return itor(x)
		}
		return itof(x)
	case ratVal, floatVal:
		return x
	case complexVal:
		if Sign(x.im) == 0 {
			return ToFloat(x.re)
		}
	}
	return unknownVal{}
}

//go:noinline
//go:registerparams
func smallInt(x *big.Int) bool {
	return false
}

//go:noinline
//go:registerparams
func Sign(x Value) int {
	return 0
}


func main() {
	v := ratVal{big.NewRat(22,7)}
	s := ToFloat(v).String()
	fmt.Printf("s=%s\n", s)
}
