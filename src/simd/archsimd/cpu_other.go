// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package archsimd

import "internal/cpu"

type ARM64Features struct{}

var ARM64 ARM64Features

// PMULL returns whether the CPU supports the PMULL feature.
//
// PMULL is defined on all GOARCHes, but will only return true on
// GOARCH arm64.
func (ARM64Features) PMULL() bool {
	return cpu.ARM64.HasPMULL
}
