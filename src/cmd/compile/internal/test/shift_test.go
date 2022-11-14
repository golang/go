// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"reflect"
	"testing"
)

// Tests shifts of zero.

//go:noinline
func ofz64l64(n uint64) int64 {
	var x int64
	return x << n
}

//go:noinline
func ofz64l32(n uint32) int64 {
	var x int64
	return x << n
}

//go:noinline
func ofz64l16(n uint16) int64 {
	var x int64
	return x << n
}

//go:noinline
func ofz64l8(n uint8) int64 {
	var x int64
	return x << n
}

//go:noinline
func ofz64r64(n uint64) int64 {
	var x int64
	return x >> n
}

//go:noinline
func ofz64r32(n uint32) int64 {
	var x int64
	return x >> n
}

//go:noinline
func ofz64r16(n uint16) int64 {
	var x int64
	return x >> n
}

//go:noinline
func ofz64r8(n uint8) int64 {
	var x int64
	return x >> n
}

//go:noinline
func ofz64ur64(n uint64) uint64 {
	var x uint64
	return x >> n
}

//go:noinline
func ofz64ur32(n uint32) uint64 {
	var x uint64
	return x >> n
}

//go:noinline
func ofz64ur16(n uint16) uint64 {
	var x uint64
	return x >> n
}

//go:noinline
func ofz64ur8(n uint8) uint64 {
	var x uint64
	return x >> n
}

//go:noinline
func ofz32l64(n uint64) int32 {
	var x int32
	return x << n
}

//go:noinline
func ofz32l32(n uint32) int32 {
	var x int32
	return x << n
}

//go:noinline
func ofz32l16(n uint16) int32 {
	var x int32
	return x << n
}

//go:noinline
func ofz32l8(n uint8) int32 {
	var x int32
	return x << n
}

//go:noinline
func ofz32r64(n uint64) int32 {
	var x int32
	return x >> n
}

//go:noinline
func ofz32r32(n uint32) int32 {
	var x int32
	return x >> n
}

//go:noinline
func ofz32r16(n uint16) int32 {
	var x int32
	return x >> n
}

//go:noinline
func ofz32r8(n uint8) int32 {
	var x int32
	return x >> n
}

//go:noinline
func ofz32ur64(n uint64) uint32 {
	var x uint32
	return x >> n
}

//go:noinline
func ofz32ur32(n uint32) uint32 {
	var x uint32
	return x >> n
}

//go:noinline
func ofz32ur16(n uint16) uint32 {
	var x uint32
	return x >> n
}

//go:noinline
func ofz32ur8(n uint8) uint32 {
	var x uint32
	return x >> n
}

//go:noinline
func ofz16l64(n uint64) int16 {
	var x int16
	return x << n
}

//go:noinline
func ofz16l32(n uint32) int16 {
	var x int16
	return x << n
}

//go:noinline
func ofz16l16(n uint16) int16 {
	var x int16
	return x << n
}

//go:noinline
func ofz16l8(n uint8) int16 {
	var x int16
	return x << n
}

//go:noinline
func ofz16r64(n uint64) int16 {
	var x int16
	return x >> n
}

//go:noinline
func ofz16r32(n uint32) int16 {
	var x int16
	return x >> n
}

//go:noinline
func ofz16r16(n uint16) int16 {
	var x int16
	return x >> n
}

//go:noinline
func ofz16r8(n uint8) int16 {
	var x int16
	return x >> n
}

//go:noinline
func ofz16ur64(n uint64) uint16 {
	var x uint16
	return x >> n
}

//go:noinline
func ofz16ur32(n uint32) uint16 {
	var x uint16
	return x >> n
}

//go:noinline
func ofz16ur16(n uint16) uint16 {
	var x uint16
	return x >> n
}

//go:noinline
func ofz16ur8(n uint8) uint16 {
	var x uint16
	return x >> n
}

//go:noinline
func ofz8l64(n uint64) int8 {
	var x int8
	return x << n
}

//go:noinline
func ofz8l32(n uint32) int8 {
	var x int8
	return x << n
}

//go:noinline
func ofz8l16(n uint16) int8 {
	var x int8
	return x << n
}

//go:noinline
func ofz8l8(n uint8) int8 {
	var x int8
	return x << n
}

//go:noinline
func ofz8r64(n uint64) int8 {
	var x int8
	return x >> n
}

//go:noinline
func ofz8r32(n uint32) int8 {
	var x int8
	return x >> n
}

//go:noinline
func ofz8r16(n uint16) int8 {
	var x int8
	return x >> n
}

//go:noinline
func ofz8r8(n uint8) int8 {
	var x int8
	return x >> n
}

//go:noinline
func ofz8ur64(n uint64) uint8 {
	var x uint8
	return x >> n
}

//go:noinline
func ofz8ur32(n uint32) uint8 {
	var x uint8
	return x >> n
}

//go:noinline
func ofz8ur16(n uint16) uint8 {
	var x uint8
	return x >> n
}

//go:noinline
func ofz8ur8(n uint8) uint8 {
	var x uint8
	return x >> n
}

func TestShiftOfZero(t *testing.T) {
	if got := ofz64l64(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz64l32(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz64l16(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz64l8(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz64r64(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz64r32(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz64r16(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz64r8(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz64ur64(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz64ur32(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz64ur16(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz64ur8(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}

	if got := ofz32l64(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz32l32(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz32l16(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz32l8(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz32r64(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz32r32(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz32r16(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz32r8(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz32ur64(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz32ur32(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz32ur16(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz32ur8(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}

	if got := ofz16l64(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz16l32(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz16l16(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz16l8(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz16r64(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz16r32(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz16r16(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz16r8(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz16ur64(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz16ur32(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz16ur16(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz16ur8(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}

	if got := ofz8l64(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz8l32(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz8l16(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz8l8(5); got != 0 {
		t.Errorf("0<<5 == %d, want 0", got)
	}
	if got := ofz8r64(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz8r32(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz8r16(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz8r8(5); got != 0 {
		t.Errorf("0>>5 == %d, want 0", got)
	}
	if got := ofz8ur64(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz8ur32(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz8ur16(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
	if got := ofz8ur8(5); got != 0 {
		t.Errorf("0>>>5 == %d, want 0", got)
	}
}

//go:noinline
func byz64l(n int64) int64 {
	return n << 0
}

//go:noinline
func byz64r(n int64) int64 {
	return n >> 0
}

//go:noinline
func byz64ur(n uint64) uint64 {
	return n >> 0
}

//go:noinline
func byz32l(n int32) int32 {
	return n << 0
}

//go:noinline
func byz32r(n int32) int32 {
	return n >> 0
}

//go:noinline
func byz32ur(n uint32) uint32 {
	return n >> 0
}

//go:noinline
func byz16l(n int16) int16 {
	return n << 0
}

//go:noinline
func byz16r(n int16) int16 {
	return n >> 0
}

//go:noinline
func byz16ur(n uint16) uint16 {
	return n >> 0
}

//go:noinline
func byz8l(n int8) int8 {
	return n << 0
}

//go:noinline
func byz8r(n int8) int8 {
	return n >> 0
}

//go:noinline
func byz8ur(n uint8) uint8 {
	return n >> 0
}

func TestShiftByZero(t *testing.T) {
	{
		var n int64 = 0x5555555555555555
		if got := byz64l(n); got != n {
			t.Errorf("%x<<0 == %x, want %x", n, got, n)
		}
		if got := byz64r(n); got != n {
			t.Errorf("%x>>0 == %x, want %x", n, got, n)
		}
	}
	{
		var n uint64 = 0xaaaaaaaaaaaaaaaa
		if got := byz64ur(n); got != n {
			t.Errorf("%x>>>0 == %x, want %x", n, got, n)
		}
	}

	{
		var n int32 = 0x55555555
		if got := byz32l(n); got != n {
			t.Errorf("%x<<0 == %x, want %x", n, got, n)
		}
		if got := byz32r(n); got != n {
			t.Errorf("%x>>0 == %x, want %x", n, got, n)
		}
	}
	{
		var n uint32 = 0xaaaaaaaa
		if got := byz32ur(n); got != n {
			t.Errorf("%x>>>0 == %x, want %x", n, got, n)
		}
	}

	{
		var n int16 = 0x5555
		if got := byz16l(n); got != n {
			t.Errorf("%x<<0 == %x, want %x", n, got, n)
		}
		if got := byz16r(n); got != n {
			t.Errorf("%x>>0 == %x, want %x", n, got, n)
		}
	}
	{
		var n uint16 = 0xaaaa
		if got := byz16ur(n); got != n {
			t.Errorf("%x>>>0 == %x, want %x", n, got, n)
		}
	}

	{
		var n int8 = 0x55
		if got := byz8l(n); got != n {
			t.Errorf("%x<<0 == %x, want %x", n, got, n)
		}
		if got := byz8r(n); got != n {
			t.Errorf("%x>>0 == %x, want %x", n, got, n)
		}
	}
	{
		var n uint8 = 0x55
		if got := byz8ur(n); got != n {
			t.Errorf("%x>>>0 == %x, want %x", n, got, n)
		}
	}
}

//go:noinline
func two64l(x int64) int64 {
	return x << 1 << 1
}

//go:noinline
func two64r(x int64) int64 {
	return x >> 1 >> 1
}

//go:noinline
func two64ur(x uint64) uint64 {
	return x >> 1 >> 1
}

//go:noinline
func two32l(x int32) int32 {
	return x << 1 << 1
}

//go:noinline
func two32r(x int32) int32 {
	return x >> 1 >> 1
}

//go:noinline
func two32ur(x uint32) uint32 {
	return x >> 1 >> 1
}

//go:noinline
func two16l(x int16) int16 {
	return x << 1 << 1
}

//go:noinline
func two16r(x int16) int16 {
	return x >> 1 >> 1
}

//go:noinline
func two16ur(x uint16) uint16 {
	return x >> 1 >> 1
}

//go:noinline
func two8l(x int8) int8 {
	return x << 1 << 1
}

//go:noinline
func two8r(x int8) int8 {
	return x >> 1 >> 1
}

//go:noinline
func two8ur(x uint8) uint8 {
	return x >> 1 >> 1
}

func TestShiftCombine(t *testing.T) {
	if got, want := two64l(4), int64(16); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := two64r(64), int64(16); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := two64ur(64), uint64(16); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := two32l(4), int32(16); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := two32r(64), int32(16); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := two32ur(64), uint32(16); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := two16l(4), int16(16); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := two16r(64), int16(16); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := two16ur(64), uint16(16); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := two8l(4), int8(16); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := two8r(64), int8(16); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := two8ur(64), uint8(16); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}

}

//go:noinline
func three64l(x int64) int64 {
	return x << 3 >> 1 << 2
}

//go:noinline
func three64ul(x uint64) uint64 {
	return x << 3 >> 1 << 2
}

//go:noinline
func three64r(x int64) int64 {
	return x >> 3 << 1 >> 2
}

//go:noinline
func three64ur(x uint64) uint64 {
	return x >> 3 << 1 >> 2
}

//go:noinline
func three32l(x int32) int32 {
	return x << 3 >> 1 << 2
}

//go:noinline
func three32ul(x uint32) uint32 {
	return x << 3 >> 1 << 2
}

//go:noinline
func three32r(x int32) int32 {
	return x >> 3 << 1 >> 2
}

//go:noinline
func three32ur(x uint32) uint32 {
	return x >> 3 << 1 >> 2
}

//go:noinline
func three16l(x int16) int16 {
	return x << 3 >> 1 << 2
}

//go:noinline
func three16ul(x uint16) uint16 {
	return x << 3 >> 1 << 2
}

//go:noinline
func three16r(x int16) int16 {
	return x >> 3 << 1 >> 2
}

//go:noinline
func three16ur(x uint16) uint16 {
	return x >> 3 << 1 >> 2
}

//go:noinline
func three8l(x int8) int8 {
	return x << 3 >> 1 << 2
}

//go:noinline
func three8ul(x uint8) uint8 {
	return x << 3 >> 1 << 2
}

//go:noinline
func three8r(x int8) int8 {
	return x >> 3 << 1 >> 2
}

//go:noinline
func three8ur(x uint8) uint8 {
	return x >> 3 << 1 >> 2
}

func TestShiftCombine3(t *testing.T) {
	if got, want := three64l(4), int64(64); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := three64ul(4), uint64(64); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := three64r(64), int64(4); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := three64ur(64), uint64(4); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := three32l(4), int32(64); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := three32ul(4), uint32(64); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := three32r(64), int32(4); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := three32ur(64), uint32(4); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := three16l(4), int16(64); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := three16ul(4), uint16(64); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := three16r(64), int16(4); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := three16ur(64), uint16(4); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := three8l(4), int8(64); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := three8ul(4), uint8(64); want != got {
		t.Errorf("4<<1<<1 == %d, want %d", got, want)
	}
	if got, want := three8r(64), int8(4); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
	if got, want := three8ur(64), uint8(4); want != got {
		t.Errorf("64>>1>>1 == %d, want %d", got, want)
	}
}

var (
	one64  int64  = 1
	one64u uint64 = 1
	one32  int32  = 1
	one32u uint32 = 1
	one16  int16  = 1
	one16u uint16 = 1
	one8   int8   = 1
	one8u  uint8  = 1
)

func TestShiftLargeCombine(t *testing.T) {
	var N uint64 = 0x8000000000000000
	if one64<<N<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one64>>N>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one64u>>N>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one32<<N<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one32>>N>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one32u>>N>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one16<<N<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one16>>N>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one16u>>N>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one8<<N<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one8>>N>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one8u>>N>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
}

func TestShiftLargeCombine3(t *testing.T) {
	var N uint64 = 0x8000000000000001
	if one64<<N>>2<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one64u<<N>>2<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one64>>N<<2>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one64u>>N<<2>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one32<<N>>2<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one32u<<N>>2<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one32>>N<<2>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one32u>>N<<2>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one16<<N>>2<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one16u<<N>>2<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one16>>N<<2>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one16u>>N<<2>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one8<<N>>2<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one8u<<N>>2<<N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one8>>N<<2>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
	if one8u>>N<<2>>N == 1 {
		t.Errorf("shift overflow mishandled")
	}
}

func TestShiftGeneric(t *testing.T) {
	for _, test := range [...]struct {
		valueWidth int
		signed     bool
		shiftWidth int
		left       bool
		f          interface{}
	}{
		{64, true, 64, true, func(n int64, s uint64) int64 { return n << s }},
		{64, true, 64, false, func(n int64, s uint64) int64 { return n >> s }},
		{64, false, 64, false, func(n uint64, s uint64) uint64 { return n >> s }},
		{64, true, 32, true, func(n int64, s uint32) int64 { return n << s }},
		{64, true, 32, false, func(n int64, s uint32) int64 { return n >> s }},
		{64, false, 32, false, func(n uint64, s uint32) uint64 { return n >> s }},
		{64, true, 16, true, func(n int64, s uint16) int64 { return n << s }},
		{64, true, 16, false, func(n int64, s uint16) int64 { return n >> s }},
		{64, false, 16, false, func(n uint64, s uint16) uint64 { return n >> s }},
		{64, true, 8, true, func(n int64, s uint8) int64 { return n << s }},
		{64, true, 8, false, func(n int64, s uint8) int64 { return n >> s }},
		{64, false, 8, false, func(n uint64, s uint8) uint64 { return n >> s }},

		{32, true, 64, true, func(n int32, s uint64) int32 { return n << s }},
		{32, true, 64, false, func(n int32, s uint64) int32 { return n >> s }},
		{32, false, 64, false, func(n uint32, s uint64) uint32 { return n >> s }},
		{32, true, 32, true, func(n int32, s uint32) int32 { return n << s }},
		{32, true, 32, false, func(n int32, s uint32) int32 { return n >> s }},
		{32, false, 32, false, func(n uint32, s uint32) uint32 { return n >> s }},
		{32, true, 16, true, func(n int32, s uint16) int32 { return n << s }},
		{32, true, 16, false, func(n int32, s uint16) int32 { return n >> s }},
		{32, false, 16, false, func(n uint32, s uint16) uint32 { return n >> s }},
		{32, true, 8, true, func(n int32, s uint8) int32 { return n << s }},
		{32, true, 8, false, func(n int32, s uint8) int32 { return n >> s }},
		{32, false, 8, false, func(n uint32, s uint8) uint32 { return n >> s }},

		{16, true, 64, true, func(n int16, s uint64) int16 { return n << s }},
		{16, true, 64, false, func(n int16, s uint64) int16 { return n >> s }},
		{16, false, 64, false, func(n uint16, s uint64) uint16 { return n >> s }},
		{16, true, 32, true, func(n int16, s uint32) int16 { return n << s }},
		{16, true, 32, false, func(n int16, s uint32) int16 { return n >> s }},
		{16, false, 32, false, func(n uint16, s uint32) uint16 { return n >> s }},
		{16, true, 16, true, func(n int16, s uint16) int16 { return n << s }},
		{16, true, 16, false, func(n int16, s uint16) int16 { return n >> s }},
		{16, false, 16, false, func(n uint16, s uint16) uint16 { return n >> s }},
		{16, true, 8, true, func(n int16, s uint8) int16 { return n << s }},
		{16, true, 8, false, func(n int16, s uint8) int16 { return n >> s }},
		{16, false, 8, false, func(n uint16, s uint8) uint16 { return n >> s }},

		{8, true, 64, true, func(n int8, s uint64) int8 { return n << s }},
		{8, true, 64, false, func(n int8, s uint64) int8 { return n >> s }},
		{8, false, 64, false, func(n uint8, s uint64) uint8 { return n >> s }},
		{8, true, 32, true, func(n int8, s uint32) int8 { return n << s }},
		{8, true, 32, false, func(n int8, s uint32) int8 { return n >> s }},
		{8, false, 32, false, func(n uint8, s uint32) uint8 { return n >> s }},
		{8, true, 16, true, func(n int8, s uint16) int8 { return n << s }},
		{8, true, 16, false, func(n int8, s uint16) int8 { return n >> s }},
		{8, false, 16, false, func(n uint8, s uint16) uint8 { return n >> s }},
		{8, true, 8, true, func(n int8, s uint8) int8 { return n << s }},
		{8, true, 8, false, func(n int8, s uint8) int8 { return n >> s }},
		{8, false, 8, false, func(n uint8, s uint8) uint8 { return n >> s }},
	} {
		fv := reflect.ValueOf(test.f)
		var args [2]reflect.Value
		for i := 0; i < test.valueWidth; i++ {
			// Build value to be shifted.
			var n int64 = 1
			for j := 0; j < i; j++ {
				n <<= 1
			}
			args[0] = reflect.ValueOf(n).Convert(fv.Type().In(0))
			for s := 0; s <= test.shiftWidth; s++ {
				args[1] = reflect.ValueOf(s).Convert(fv.Type().In(1))

				// Compute desired result. We're testing variable shifts
				// assuming constant shifts are correct.
				r := n
				var op string
				switch {
				case test.left:
					op = "<<"
					for j := 0; j < s; j++ {
						r <<= 1
					}
					switch test.valueWidth {
					case 32:
						r = int64(int32(r))
					case 16:
						r = int64(int16(r))
					case 8:
						r = int64(int8(r))
					}
				case test.signed:
					op = ">>"
					switch test.valueWidth {
					case 32:
						r = int64(int32(r))
					case 16:
						r = int64(int16(r))
					case 8:
						r = int64(int8(r))
					}
					for j := 0; j < s; j++ {
						r >>= 1
					}
				default:
					op = ">>>"
					for j := 0; j < s; j++ {
						r = int64(uint64(r) >> 1)
					}
				}

				// Call function.
				res := fv.Call(args[:])[0].Convert(reflect.ValueOf(r).Type())

				if res.Int() != r {
					t.Errorf("%s%dx%d(%x,%x)=%x, want %x", op, test.valueWidth, test.shiftWidth, n, s, res.Int(), r)
				}
			}
		}
	}
}

var shiftSink64 int64

func BenchmarkShiftArithmeticRight(b *testing.B) {
	x := shiftSink64
	for i := 0; i < b.N; i++ {
		x = x >> (i & 63)
	}
	shiftSink64 = x
}

//go:noinline
func incorrectRotate1(x, c uint64) uint64 {
	// This should not compile to a rotate instruction.
	return x<<c | x>>(64-c)
}

//go:noinline
func incorrectRotate2(x uint64) uint64 {
	var c uint64 = 66
	// This should not compile to a rotate instruction.
	return x<<c | x>>(64-c)
}

func TestIncorrectRotate(t *testing.T) {
	if got := incorrectRotate1(1, 66); got != 0 {
		t.Errorf("got %x want 0", got)
	}
	if got := incorrectRotate2(1); got != 0 {
		t.Errorf("got %x want 0", got)
	}
}
