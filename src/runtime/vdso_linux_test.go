// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || amd64 || arm || arm64 || loong64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x)

package runtime_test

import (
	"runtime"
	"testing"
)

// DT_GNU_HASH hash function.
func gnuHash(s string) uint32 {
	h := uint32(5381)
	for _, r := range s {
		h = (h << 5) + h + uint32(r)
	}
	return h
}

// DT_HASH hash function.
func symHash(s string) uint32 {
	var h, g uint32
	for _, r := range s {
		h = (h << 4) + uint32(r)
		g = h & 0xf0000000
		if g != 0 {
			h ^= g >> 24
		}
		h &^= g
	}
	return h
}

func TestVDSOHash(t *testing.T) {
	for _, sym := range runtime.VDSOSymbolKeys() {
		name := sym.Name()
		t.Run(name, func(t *testing.T) {
			want := symHash(name)
			if sym.SymHash() != want {
				t.Errorf("SymHash got %#x want %#x", sym.SymHash(), want)
			}

			want = gnuHash(name)
			if sym.GNUHash() != want {
				t.Errorf("GNUHash got %#x want %#x", sym.GNUHash(), want)
			}
		})
	}
}
