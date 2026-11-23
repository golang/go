// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"internal/buildcfg"
	"strings"
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

const SimdPkg = "simd"
const SimdBridgePkg = "simd/internal/bridge"

// Private @-prefixed import aliases used for midway-generated references so
// injected imports cannot collide with user identifiers, imports, or locals.
const simdPkgAlias = "@simd"
const archPkg = "bridge"
const archPkgAlias = "@bridge"
const vectorSizeFn = "VectorBits"
const emulatedFn = "Emulated"

// IsSpecializedName reports whether name is the name midway gave to a
// size-specialized copy of a package-level declaration, such as "F@simd128".
// "@" can't appear in a Go identifier, so no user declaration has such a name.
func IsSpecializedName(name string) bool {
	return strings.Contains(name, "@simd")
}

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
