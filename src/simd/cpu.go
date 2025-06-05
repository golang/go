// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

// The build condition == if the experiment is not on, cmd/api TestCheck will see this and complain
// see also go/doc/comment, where "simd" is inserted to the package list of the experiment is not on.

package simd

import "internal/cpu"

func HasAVX512BW() bool {
	return cpu.X86.HasAVX512BW
}

func HasAVX512VL() bool {
	return cpu.X86.HasAVX512VL
}
