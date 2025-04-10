// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Asmgen generates math/big assembly.
//
// Usage:
//
//	cd go/src/math/big
//	go test ./internal/asmgen -generate
//
// Or:
//
//	go generate math/big
package asmgen

var arches = []*Arch{
	Arch386,
	ArchAMD64,
	ArchARM,
	ArchARM64,
	ArchLoong64,
	ArchMIPS,
	ArchMIPS64x,
	ArchPPC64x,
	ArchRISCV64,
	ArchS390X,
}

// generate returns the file name and content of the generated assembly for the given architecture.
func generate(arch *Arch) (file string, data []byte) {
	file = "arith_" + arch.Name + ".s"
	a := NewAsm(arch)
	addOrSubVV(a, "addVV")
	addOrSubVV(a, "subVV")
	return file, a.out.Bytes()
}
