// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"math/bits"
	"testing"
)

func BenchmarkSwitch8Predictable(b *testing.B) {
	benchmarkSwitch8(b, true)
}
func BenchmarkSwitch8Unpredictable(b *testing.B) {
	benchmarkSwitch8(b, false)
}
func benchmarkSwitch8(b *testing.B, predictable bool) {
	n := 0
	rng := newRNG()
	for i := 0; i < b.N; i++ {
		rng = rng.next(predictable)
		switch rng.value() & 7 {
		case 0:
			n += 1
		case 1:
			n += 2
		case 2:
			n += 3
		case 3:
			n += 4
		case 4:
			n += 5
		case 5:
			n += 6
		case 6:
			n += 7
		case 7:
			n += 8
		}
	}
	sink = n
}

func BenchmarkSwitch32Predictable(b *testing.B) {
	benchmarkSwitch32(b, true)
}
func BenchmarkSwitch32Unpredictable(b *testing.B) {
	benchmarkSwitch32(b, false)
}
func benchmarkSwitch32(b *testing.B, predictable bool) {
	n := 0
	rng := newRNG()
	for i := 0; i < b.N; i++ {
		rng = rng.next(predictable)
		switch rng.value() & 31 {
		case 0, 1, 2:
			n += 1
		case 4, 5, 6:
			n += 2
		case 8, 9, 10:
			n += 3
		case 12, 13, 14:
			n += 4
		case 16, 17, 18:
			n += 5
		case 20, 21, 22:
			n += 6
		case 24, 25, 26:
			n += 7
		case 28, 29, 30:
			n += 8
		default:
			n += 9
		}
	}
	sink = n
}

func BenchmarkSwitchStringPredictable(b *testing.B) {
	benchmarkSwitchString(b, true)
}
func BenchmarkSwitchStringUnpredictable(b *testing.B) {
	benchmarkSwitchString(b, false)
}
func benchmarkSwitchString(b *testing.B, predictable bool) {
	a := []string{
		"foo",
		"foo1",
		"foo22",
		"foo333",
		"foo4444",
		"foo55555",
		"foo666666",
		"foo7777777",
	}
	n := 0
	rng := newRNG()
	for i := 0; i < b.N; i++ {
		rng = rng.next(predictable)
		switch a[rng.value()&7] {
		case "foo":
			n += 1
		case "foo1":
			n += 2
		case "foo22":
			n += 3
		case "foo333":
			n += 4
		case "foo4444":
			n += 5
		case "foo55555":
			n += 6
		case "foo666666":
			n += 7
		case "foo7777777":
			n += 8
		}
	}
	sink = n
}

func BenchmarkSwitchTypePredictable(b *testing.B) {
	benchmarkSwitchType(b, true)
}
func BenchmarkSwitchTypeUnpredictable(b *testing.B) {
	benchmarkSwitchType(b, false)
}
func benchmarkSwitchType(b *testing.B, predictable bool) {
	a := []any{
		int8(1),
		int16(2),
		int32(3),
		int64(4),
		uint8(5),
		uint16(6),
		uint32(7),
		uint64(8),
	}
	n := 0
	rng := newRNG()
	for i := 0; i < b.N; i++ {
		rng = rng.next(predictable)
		switch a[rng.value()&7].(type) {
		case int8:
			n += 1
		case int16:
			n += 2
		case int32:
			n += 3
		case int64:
			n += 4
		case uint8:
			n += 5
		case uint16:
			n += 6
		case uint32:
			n += 7
		case uint64:
			n += 8
		}
	}
	sink = n
}

func BenchmarkSwitchInterfaceTypePredictable(b *testing.B) {
	benchmarkSwitchInterfaceType(b, true)
}
func BenchmarkSwitchInterfaceTypeUnpredictable(b *testing.B) {
	benchmarkSwitchInterfaceType(b, false)
}

type SI0 interface {
	si0()
}
type ST0 struct {
}

func (ST0) si0() {
}

type SI1 interface {
	si1()
}
type ST1 struct {
}

func (ST1) si1() {
}

type SI2 interface {
	si2()
}
type ST2 struct {
}

func (ST2) si2() {
}

type SI3 interface {
	si3()
}
type ST3 struct {
}

func (ST3) si3() {
}

type SI4 interface {
	si4()
}
type ST4 struct {
}

func (ST4) si4() {
}

type SI5 interface {
	si5()
}
type ST5 struct {
}

func (ST5) si5() {
}

type SI6 interface {
	si6()
}
type ST6 struct {
}

func (ST6) si6() {
}

type SI7 interface {
	si7()
}
type ST7 struct {
}

func (ST7) si7() {
}

func benchmarkSwitchInterfaceType(b *testing.B, predictable bool) {
	a := []any{
		ST0{},
		ST1{},
		ST2{},
		ST3{},
		ST4{},
		ST5{},
		ST6{},
		ST7{},
	}
	n := 0
	rng := newRNG()
	for i := 0; i < b.N; i++ {
		rng = rng.next(predictable)
		switch a[rng.value()&7].(type) {
		case SI0:
			n += 1
		case SI1:
			n += 2
		case SI2:
			n += 3
		case SI3:
			n += 4
		case SI4:
			n += 5
		case SI5:
			n += 6
		case SI6:
			n += 7
		case SI7:
			n += 8
		}
	}
	sink = n
}

// A simple random number generator used to make switches conditionally predictable.
type rng uint64

func newRNG() rng {
	return 1
}
func (r rng) next(predictable bool) rng {
	if predictable {
		return r + 1
	}
	return rng(bits.RotateLeft64(uint64(r), 13) * 0x3c374d)
}
func (r rng) value() uint64 {
	return uint64(r)
}
