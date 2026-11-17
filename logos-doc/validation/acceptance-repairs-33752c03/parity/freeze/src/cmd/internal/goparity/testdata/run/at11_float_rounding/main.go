// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/big"
	"os"
)

var failed bool

var vx, vy, vz float64
var fx, fy, fz float32

func init() {
	vx = 1 + math.Ldexp(1, -27)
	vy = 1 + math.Ldexp(1, -27)
	vz = -1
	fx = float32(1 + math.Ldexp(1, -12))
	fy = float32(1 + math.Ldexp(1, -12))
	fz = -1
}

func b53(f float64) *big.Float {
	return new(big.Float).SetPrec(53).SetMode(big.ToNearestEven).SetFloat64(f)
}

func b24(f float32) *big.Float {
	return new(big.Float).SetPrec(24).SetMode(big.ToNearestEven).SetFloat64(float64(f))
}

func refAdd64(x, y, z float64) float64 {
	p := new(big.Float).SetPrec(53).SetMode(big.ToNearestEven).Mul(b53(x), b53(y))
	s := new(big.Float).SetPrec(53).SetMode(big.ToNearestEven).Add(p, b53(z))
	f, _ := s.Float64()
	return f
}

func refSub64(x, y, z float64) float64 {
	p := new(big.Float).SetPrec(53).SetMode(big.ToNearestEven).Mul(b53(x), b53(y))
	s := new(big.Float).SetPrec(53).SetMode(big.ToNearestEven).Sub(p, b53(z))
	f, _ := s.Float64()
	return f
}

func refAdd32(x, y, z float32) float32 {
	p := new(big.Float).SetPrec(24).SetMode(big.ToNearestEven).Mul(b24(x), b24(y))
	s := new(big.Float).SetPrec(24).SetMode(big.ToNearestEven).Add(p, b24(z))
	f, _ := s.Float32()
	return f
}

func refSub32(x, y, z float32) float32 {
	p := new(big.Float).SetPrec(24).SetMode(big.ToNearestEven).Mul(b24(x), b24(y))
	s := new(big.Float).SetPrec(24).SetMode(big.ToNearestEven).Sub(p, b24(z))
	f, _ := s.Float32()
	return f
}

func exactOK(f float64) bool {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return false
	}
	a := math.Abs(f)
	return a == 0 || (a >= 2.2250738585072014e-308 && a <= math.MaxFloat64)
}

func exactOK32(f float32) bool {
	if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
		return false
	}
	a := float64(f)
	if a < 0 {
		a = -a
	}
	return a == 0 || (a >= 1.1754943508222875e-38 && a <= float64(float32(math.MaxFloat32)))
}

func check64(tag string, x, y, z, got, want float64) {
	if math.Float64bits(got) != math.Float64bits(want) {
		fmt.Printf("LAW %s x=%d y=%d z=%d got=%d want=%d\n", tag, math.Float64bits(x), math.Float64bits(y), math.Float64bits(z), math.Float64bits(got), math.Float64bits(want))
		failed = true
	}
}

func check32(tag string, x, y, z, got, want float32) {
	if math.Float32bits(got) != math.Float32bits(want) {
		fmt.Printf("LAW %s x=%d y=%d z=%d got=%d want=%d\n", tag, math.Float32bits(x), math.Float32bits(y), math.Float32bits(z), math.Float32bits(got), math.Float32bits(want))
		failed = true
	}
}

func nanIn64(x, y, z float64) bool {
	return math.IsNaN(x) || math.IsNaN(y) || math.IsNaN(z)
}

func nanIn32(x, y, z float32) bool {
	return math.IsNaN(float64(x)) || math.IsNaN(float64(y)) || math.IsNaN(float64(z))
}

func trial64(x, y, z float64, tag string) {
	fmt.Printf("case64 %s x=%d y=%d z=%d\n", tag, math.Float64bits(x), math.Float64bits(y), math.Float64bits(z))
	add := float64(x*y) + z
	sub := float64(x*y) - z
	fmt.Printf("conv64 %s add=%d sub=%d\n", tag, math.Float64bits(add), math.Float64bits(sub))
	fmt.Printf("raw64 %s raw=%d\n", tag, math.Float64bits(x*y+z))
	if nanIn64(x, y, z) {
		if !math.IsNaN(add) || !math.IsNaN(sub) {
			fmt.Printf("LAW %s nannonprop\n", tag)
			failed = true
		}
		return
	}
	p := x * y
	if exactOK(p) && exactOK(add) && exactOK(sub) {
		check64(tag+"-add", x, y, z, add, refAdd64(x, y, z))
		check64(tag+"-sub", x, y, z, sub, refSub64(x, y, z))
	}
}

func trial32(x, y, z float32, tag string) {
	fmt.Printf("case32 %s x=%d y=%d z=%d\n", tag, math.Float32bits(x), math.Float32bits(y), math.Float32bits(z))
	add := float32(x*y) + z
	sub := float32(x*y) - z
	fmt.Printf("conv32 %s add=%d sub=%d\n", tag, math.Float32bits(add), math.Float32bits(sub))
	fmt.Printf("raw32 %s raw=%d\n", tag, math.Float32bits(x*y+z))
	if nanIn32(x, y, z) {
		if !math.IsNaN(float64(add)) || !math.IsNaN(float64(sub)) {
			fmt.Printf("LAW %s nannonprop\n", tag)
			failed = true
		}
		return
	}
	p := x * y
	if exactOK32(p) && exactOK32(add) && exactOK32(sub) {
		check32(tag+"-add", x, y, z, add, refAdd32(x, y, z))
		check32(tag+"-sub", x, y, z, sub, refSub32(x, y, z))
	}
}

func step(s *uint64) uint64 {
	*s ^= *s << 13
	*s ^= *s >> 7
	*s ^= *s << 17
	return *s
}

func gen64(s *uint64) float64 {
	m := step(s) & ((1 << 52) - 1)
	e := uint64(1023-30) + step(s)%61
	return math.Float64frombits(m | (e << 52))
}

func gen32(s *uint64) float32 {
	m := uint32(step(s) & ((1 << 23) - 1))
	e := uint32(127-15) + uint32(step(s)%31)
	return math.Float32frombits(m | (e << 23))
}

func main() {
	trial64(vx, vy, vz, "killer")
	trial32(fx, fy, fz, "killer32")
	trial64(math.SmallestNonzeroFloat64, 1.5, 0, "submin")
	trial64(5e-324, 5e-324, 1e-300, "subadd")
	trial64(math.Inf(1), 2, 3, "inf1")
	trial64(math.Inf(-1), math.Inf(1), 5, "inf2")
	trial64(math.Inf(1), 0, 1, "infzero")
	trial64(math.NaN(), 1, 2, "nan1")
	trial64(1, math.NaN(), math.Inf(1), "nan2")
	trial32(float32(math.SmallestNonzeroFloat32), 1.5, 0, "submin32")
	trial32(float32(math.Inf(1)), 2, 3, "inf32")
	trial32(float32(math.NaN()), 1, 2, "nan32")
	s := uint64(0x9e3779b97f4a7c15)
	for range 200 {
		x := gen64(&s)
		y := gen64(&s)
		z := gen64(&s)
		trial64(x, y, z, "seed")
	}
	for range 200 {
		x := gen32(&s)
		y := gen32(&s)
		z := gen32(&s)
		trial32(x, y, z, "seed32")
	}
	if failed {
		os.Exit(1)
	}
	fmt.Println("ok")
}
