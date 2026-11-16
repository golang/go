// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd

import (
	"internal/cpu"
	"simd/archsimd"
	"testing"
)

func TestAMD64WithoutFMA(t *testing.T) {
	if !archsimd.X86.AVX() {
		t.Skip("requires AVX")
	}
	saved := cpu.X86.HasFMA
	cpu.X86.HasFMA = false
	defer func() { cpu.X86.HasFMA = saved }()
	if archsimd.X86.FMA() {
		t.Skip("FMA is guaranteed by GOAMD64")
	}
	size, allFeatureSize, arch := archMaxVectorSize()
	_, emulated, _ := configure(size, allFeatureSize, arch, "")
	if !emulated {
		t.Fatal("hardware SIMD selected without FMA")
	}
}
