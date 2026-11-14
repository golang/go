// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps_test

import (
	"internal/abi"
	"internal/runtime/maps"
	"runtime"
	"testing"
)

func TestCtrlGroupZeroEmpty(t *testing.T) {
	for _, portable := range []bool{false, true} {
		got := ctrlGroupMasks(t, 0, 0, portable)
		if got != [4]uint8{0, 0xff, 0xff, 0} {
			t.Errorf("zero control word, portable=%v: got masks %x, want [00 ff ff 00]", portable, got)
		}
	}
	if maps.CtrlGroupAnyFull(0) {
		t.Error("zero control word contains full slots")
	}
	if got := maps.CtrlGroupSetEmpty(^maps.CtrlGroup(0)); got != 0 {
		t.Errorf("setEmpty: got %#x, want zero", got)
	}
}

func TestCtrlGroupAccess(t *testing.T) {
	for slot := uintptr(0); slot < abi.MapGroupSlots; slot++ {
		for value := uint16(0); value < 256; value++ {
			before := maps.CtrlGroup(0x123456789abcdef0)
			got, read := maps.CtrlGroupSet(before, slot, uint8(value))
			shift := slot * 8
			want := before&^(maps.CtrlGroup(0xff)<<shift) | maps.CtrlGroup(value)<<shift
			if got != want || read != uint8(value) {
				t.Fatalf("slot %d value %#x: group=%#x read=%#x, want group=%#x", slot, value, got, read, want)
			}
		}
	}
}

func TestH2ControlByte(t *testing.T) {
	for _, high := range []uintptr{0, 1 << 8, 1 << 16, ^uintptr(0) &^ 0xff} {
		for low := uintptr(0); low < 256; low++ {
			hash := high | low
			got := uint8(maps.H2(hash))
			want := uint8(low&0x7f) | 0x80
			if got != want {
				t.Fatalf("h2(%#x) = %#x, want %#x", hash, got, want)
			}
		}
	}
}

// Adjacent full slots exercise the byte-subtraction borrow in the generic
// matcher. Empty and deleted slots must never be returned as hash matches.
func TestCtrlGroupMatchers(t *testing.T) {
	states := []uint8{uint8(maps.CtrlEmpty), uint8(maps.CtrlDeleted)}
	for hash := uintptr(0); hash < 128; hash++ {
		states = append(states, uint8(maps.H2(hash)))
	}
	for _, portable := range []bool{false, true} {
		for slot := uintptr(0); slot < abi.MapGroupSlots-1; slot++ {
			for _, first := range states {
				for _, second := range states {
					var group maps.CtrlGroup
					for i := uintptr(0); i < abi.MapGroupSlots; i++ {
						group |= maps.CtrlGroup(maps.CtrlEmpty) << (8 * i)
					}
					group &^= maps.CtrlGroup(0xffff) << (8 * slot)
					group |= maps.CtrlGroup(first)<<uint(8*slot) | maps.CtrlGroup(second)<<uint(8*(slot+1))
					checkCtrlGroup(t, group, portable)
				}
			}
		}
		for hash := uintptr(0); hash < 128; hash++ {
			var repeated, alternating maps.CtrlGroup
			for slot := uintptr(0); slot < abi.MapGroupSlots; slot++ {
				repeated |= maps.CtrlGroup(maps.H2(hash)) << (8 * slot)
				alternating |= maps.CtrlGroup(maps.H2(hash^(slot&1))) << (8 * slot)
			}
			checkCtrlGroup(t, repeated, portable)
			checkCtrlGroup(t, alternating, portable)
		}
	}
}

func checkCtrlGroup(t *testing.T, group maps.CtrlGroup, portable bool) {
	var empty, emptyOrDeleted, full uint8
	var hashes [128]uint8
	for slot := uintptr(0); slot < abi.MapGroupSlots; slot++ {
		control := uint8(group >> (8 * slot))
		bit := uint8(1) << slot
		switch control {
		case uint8(maps.CtrlEmpty):
			empty |= bit
			emptyOrDeleted |= bit
		case uint8(maps.CtrlDeleted):
			emptyOrDeleted |= bit
		default:
			full |= bit
			hashes[control&0x7f] |= bit
		}
	}
	if got := maps.CtrlGroupAnyFull(group); got != (full != 0) {
		t.Fatalf("group=%#x anyFull=%v, want %v", group, got, full != 0)
	}
	for hash := uintptr(0); hash < 128; hash++ {
		got := ctrlGroupMasks(t, group, hash, portable)
		if got[1] != empty || got[2] != emptyOrDeleted || got[3] != full {
			t.Fatalf("group=%#x hash=%#x portable=%v: state masks=%x, want [%02x %02x %02x]", group, hash, portable, got[1:], empty, emptyOrDeleted, full)
		}
		want := hashes[hash]
		if got[0]&want != want || got[0]&emptyOrDeleted != 0 || (want == 0 && got[0] != 0) {
			t.Fatalf("group=%#x hash=%#x portable=%v: hash mask=%02x, required=%02x excluded=%02x", group, hash, portable, got[0], want, emptyOrDeleted)
		}
		// The ARM64 arithmetic matcher and AMD64 intrinsic are exact.
		if (runtime.GOARCH == "arm64" || (maps.BitsetIsPacked() && !portable)) && got[0] != want {
			t.Fatalf("group=%#x hash=%#x portable=%v: exact hash mask=%02x, want %02x", group, hash, portable, got[0], want)
		}
	}
}

// Convert both packed intrinsic results and byte-wise portable results
// into one bit per slot so that the expected masks are architecture independent.
func ctrlGroupMasks(t *testing.T, group maps.CtrlGroup, hash uintptr, portable bool) [4]uint8 {
	matches := maps.CtrlGroupMatch(group, maps.H2(hash), portable)
	var masks [4]uint8
	for i, match := range matches {
		if maps.BitsetIsPacked() && !portable {
			if match&^uint64(0xff) != 0 {
				t.Fatalf("group=%#x hash=%#x matcher=%d: packed mask=%#x has bits outside the group", group, hash, i, match)
			}
			masks[i] = uint8(match)
			continue
		}
		if match&^uint64(0x8080808080808080) != 0 {
			t.Fatalf("group=%#x hash=%#x matcher=%d portable=%v: byte-wise mask=%#x has bits outside slot markers", group, hash, i, portable, match)
		}
		for slot := uintptr(0); slot < abi.MapGroupSlots; slot++ {
			if match&(uint64(0x80)<<(8*slot)) != 0 {
				masks[i] |= uint8(1) << slot
			}
		}
	}
	return masks
}
