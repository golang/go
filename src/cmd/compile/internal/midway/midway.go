// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"internal/buildcfg"
)

func rewriteSizes() []int {
	switch buildcfg.GOARCH {
	case "wasm":
		return []int{0, 128}
	case "amd64":
		return []int{0, 128, 256, 512}
	case "arm64":
		return []int{0, 128} // this will change for SVE and cannot just be a size-based choice.
	}
	return nil
}

const simdPkg = "simd"
const archFullPkg = "simd/internal/bridge"
const archPkg = "bridge"
const vectorSizeFn = "VectorBitSize"
const emulatedFn = "Emulated"

func isSimdTypeName(s string) bool {
	switch s {
	case "Int8s", "Int16s", "Int32s", "Int64s",
		"Uint8s", "Uint16s", "Uint32s", "Uint64s",
		"Mask8s", "Mask16s", "Mask32s", "Mask64s",
		"Float32s", "Float64s":
		return true
	}
	return false
}
