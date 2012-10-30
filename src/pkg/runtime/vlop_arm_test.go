// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import "testing"

// arm soft division benchmarks adapted from
// http://ridiculousfish.com/files/division_benchmarks.tar.gz

const numeratorsSize = 1 << 21

var numerators = randomNumerators()

type randstate struct {
	hi, lo uint32
}

func (r *randstate) rand() uint32 {
	r.hi = r.hi<<16 + r.hi>>16
	r.hi += r.lo
	r.lo += r.hi
	return r.hi
}

func randomNumerators() []uint32 {
	numerators := make([]uint32, numeratorsSize)
	random := &randstate{2147483563, 2147483563 ^ 0x49616E42}
	for i := range numerators {
		numerators[i] = random.rand()
	}
	return numerators
}

func bmUint32Div(divisor uint32, b *testing.B) {
	var sum uint32
	for i := 0; i < b.N; i++ {
		sum += numerators[i&(numeratorsSize-1)] / divisor
	}
}

func BenchmarkUint32Div7(b *testing.B)         { bmUint32Div(7, b) }
func BenchmarkUint32Div37(b *testing.B)        { bmUint32Div(37, b) }
func BenchmarkUint32Div123(b *testing.B)       { bmUint32Div(123, b) }
func BenchmarkUint32Div763(b *testing.B)       { bmUint32Div(763, b) }
func BenchmarkUint32Div1247(b *testing.B)      { bmUint32Div(1247, b) }
func BenchmarkUint32Div9305(b *testing.B)      { bmUint32Div(9305, b) }
func BenchmarkUint32Div13307(b *testing.B)     { bmUint32Div(13307, b) }
func BenchmarkUint32Div52513(b *testing.B)     { bmUint32Div(52513, b) }
func BenchmarkUint32Div60978747(b *testing.B)  { bmUint32Div(60978747, b) }
func BenchmarkUint32Div106956295(b *testing.B) { bmUint32Div(106956295, b) }

func bmUint32Mod(divisor uint32, b *testing.B) {
	var sum uint32
	for i := 0; i < b.N; i++ {
		sum += numerators[i&(numeratorsSize-1)] % divisor
	}
}

func BenchmarkUint32Mod7(b *testing.B)         { bmUint32Mod(7, b) }
func BenchmarkUint32Mod37(b *testing.B)        { bmUint32Mod(37, b) }
func BenchmarkUint32Mod123(b *testing.B)       { bmUint32Mod(123, b) }
func BenchmarkUint32Mod763(b *testing.B)       { bmUint32Mod(763, b) }
func BenchmarkUint32Mod1247(b *testing.B)      { bmUint32Mod(1247, b) }
func BenchmarkUint32Mod9305(b *testing.B)      { bmUint32Mod(9305, b) }
func BenchmarkUint32Mod13307(b *testing.B)     { bmUint32Mod(13307, b) }
func BenchmarkUint32Mod52513(b *testing.B)     { bmUint32Mod(52513, b) }
func BenchmarkUint32Mod60978747(b *testing.B)  { bmUint32Mod(60978747, b) }
func BenchmarkUint32Mod106956295(b *testing.B) { bmUint32Mod(106956295, b) }
