// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || amd64 || arm || arm64 || loong64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x)

package runtime

type VDSOSymbolKey vdsoSymbolKey

func (v VDSOSymbolKey) Name() string {
	return v.name
}

func (v VDSOSymbolKey) SymHash() uint32 {
	return v.symHash
}

func (v VDSOSymbolKey) GNUHash() uint32 {
	return v.gnuHash
}

func VDSOSymbolKeys() []VDSOSymbolKey {
	keys := make([]VDSOSymbolKey, 0, len(vdsoSymbolKeys))
	for _, k := range vdsoSymbolKeys {
		keys = append(keys, VDSOSymbolKey(k))
	}
	return keys
}
