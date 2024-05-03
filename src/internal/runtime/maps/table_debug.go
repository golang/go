// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package maps implements Go's builtin map type.
package maps

import (
	"internal/abi"
	"unsafe"
)

const debugLog = false

func (t *table) checkInvariants() {
	if !debugLog {
		return
	}

	// For every non-empty slot, verify we can retrieve the key using Get.
	// Count the number of used and deleted slots.
	var used uint64
	var deleted uint64
	var empty uint64
	for i := uint64(0); i <= t.groups.lengthMask; i++ {
		g := t.groups.group(i)
		for j := uint32(0); j < abi.SwissMapGroupSlots; j++ {
			c := g.ctrls().get(j)
			switch {
			case c == ctrlDeleted:
				deleted++
			case c == ctrlEmpty:
				empty++
			default:
				used++

				key := g.key(j)

				// Can't lookup keys that don't compare equal
				// to themselves (e.g., NaN).
				if !t.typ.Key.Equal(key, key) {
					continue
				}

				if _, ok := t.Get(key); !ok {
					hash := t.typ.Hasher(key, t.seed)
					print("invariant failed: slot(", i, "/", j, "): key ")
					dump(key, t.typ.Key.Size_)
					print(" not found [hash=", hash, ", h2=", h2(hash), " h1=", h1(hash), "]\n")
					t.Print()
					panic("invariant failed: slot: key not found")
				}
			}
		}
	}

	if used != t.used {
		print("invariant failed: found ", used, " used slots, but used count is ", t.used, "\n")
		t.Print()
		panic("invariant failed: found mismatched used slot count")
	}

	growthLeft := (t.capacity*maxAvgGroupLoad)/abi.SwissMapGroupSlots - t.used - deleted
	if growthLeft != t.growthLeft {
		print("invariant failed: found ", t.growthLeft, " growthLeft, but expected ", growthLeft, "\n")
		t.Print()
		panic("invariant failed: found mismatched growthLeft")
	}
	if deleted != t.tombstones() {
		print("invariant failed: found ", deleted, " tombstones, but expected ", t.tombstones(), "\n")
		t.Print()
		panic("invariant failed: found mismatched tombstones")
	}

	if empty == 0 {
		print("invariant failed: found no empty slots (violates probe invariant)\n")
		t.Print()
		panic("invariant failed: found no empty slots (violates probe invariant)")
	}
}

func (t *table) Print() {
	print(`table{
	seed: `, t.seed, `
	capacity: `, t.capacity, `
	used: `, t.used, `
	growthLeft: `, t.growthLeft, `
	groups:
`)

	for i := uint64(0); i <= t.groups.lengthMask; i++ {
		print("\t\tgroup ", i, "\n")

		g := t.groups.group(i)
		ctrls := g.ctrls()
		for j := uint32(0); j < abi.SwissMapGroupSlots; j++ {
			print("\t\t\tslot ", j, "\n")

			c := ctrls.get(j)
			print("\t\t\t\tctrl ", c)
			switch c {
			case ctrlEmpty:
				print(" (empty)\n")
			case ctrlDeleted:
				print(" (deleted)\n")
			default:
				print("\n")
			}

			print("\t\t\t\tkey  ")
			dump(g.key(j), t.typ.Key.Size_)
			println("")
			print("\t\t\t\telem ")
			dump(g.elem(j), t.typ.Elem.Size_)
			println("")
		}
	}
}

// TODO(prattmic): not in hex because print doesn't have a way to print in hex
// outside the runtime.
func dump(ptr unsafe.Pointer, size uintptr) {
	for size > 0 {
		print(*(*byte)(ptr), " ")
		ptr = unsafe.Pointer(uintptr(ptr) + 1)
		size--
	}
}
