// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64latelower

import (
	"internal/buildcfg"

	"cmd/compile/internal/ssa"
)

// hasAVX2 reports whether AVX2 instructions may be used for v, either because
// GOAMD64 requires AVX2 or because the cpufeatures pass proved that v's block
// only runs on CPUs with AVX2.
func hasAVX2(v *ssa.Value) bool {
	return buildcfg.GOAMD64 >= 3 || v.Block.CPUfeatures.HasFeature(ssa.CPUavx2)
}
