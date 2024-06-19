// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego && (386 || amd64 || arm || arm64 || loong64 || ppc64 || ppc64le || riscv64 || s390x)

package bigmod

import "internal/cpu"

// amd64 assembly uses ADCX/ADOX/MULX if ADX is available to run two carry
// chains in the flags in parallel across the whole operation, and aggressively
// unrolls loops. arm64 processes four words at a time.
//
// It's unclear why the assembly for all other architectures, as well as for
// amd64 without ADX, perform better than the compiler output.
// TODO(filippo): file cmd/compile performance issue.

var supportADX = cpu.X86.HasADX && cpu.X86.HasBMI2

//go:noescape
func addMulVVW1024(z, x *uint, y uint) (c uint)

//go:noescape
func addMulVVW1536(z, x *uint, y uint) (c uint)

//go:noescape
func addMulVVW2048(z, x *uint, y uint) (c uint)
