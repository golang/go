// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

import "internal/goarch"

type InterfaceSwitch struct {
	Cache  *InterfaceSwitchCache
	NCases int

	// Array of NCases elements.
	// Each case must be a non-empty interface type.
	Cases [1]*InterfaceType
}

type InterfaceSwitchCache struct {
	Mask    uintptr                      // mask for index. Must be a power of 2 minus 1
	Entries [1]InterfaceSwitchCacheEntry // Mask+1 entries total
}

type InterfaceSwitchCacheEntry struct {
	// type of source value (a *Type)
	Typ uintptr
	// case # to dispatch to
	Case int
	// itab to use for resulting case variable (a *runtime.itab)
	Itab uintptr
}

func UseInterfaceSwitchCache(arch goarch.ArchFamilyType) bool {
	// We need an atomic load instruction to make the cache multithreaded-safe.
	// (AtomicLoadPtr needs to be implemented in cmd/compile/internal/ssa/_gen/ARCH.rules.)
	switch arch {
	case goarch.AMD64, goarch.ARM64, goarch.LOONG64, goarch.MIPS, goarch.MIPS64, goarch.PPC64, goarch.RISCV64, goarch.S390X:
		return true
	default:
		return false
	}
}

type TypeAssert struct {
	Cache   *TypeAssertCache
	Inter   *InterfaceType
	CanFail bool
}
type TypeAssertCache struct {
	Mask    uintptr
	Entries [1]TypeAssertCacheEntry
}
type TypeAssertCacheEntry struct {
	// type of source value (a *runtime._type)
	Typ uintptr
	// itab to use for result (a *runtime.itab)
	// nil if CanFail is set and conversion would fail.
	Itab uintptr
}
