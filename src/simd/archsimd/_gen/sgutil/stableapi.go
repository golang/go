// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import "strings"

// NonSpecAPI reports whether the exported declaration recv.name (or name, for
// a package-level declaration) in package pkg is one of the few APIs that
// spec doesn't describe but that may still be generally available. pkg is the
// package name, "simd" or "archsimd". Any other generally available API must
// come from spec.
//
// Each of these APIs may be either generally available or experimental. Since
// spec doesn't describe them, the code that defines each one decides which,
// so the phase check allows them in files of either phase. For example,
// simdgen puts a CPU feature query in cpu.go, which is generally available, if
// some generally available operation needs that feature, and in cpu_exp.go,
// which is experimental, otherwise.
func NonSpecAPI(pkg, recv, name string) bool {
	switch pkg {
	case "archsimd":
		switch recv {
		case "":
			// The CPU feature sets: their types, and the variables that
			// hold them.
			return name == "X86Features" || name == "X86" || name == "ARM64Features" || name == "ARM64"
		case "X86Features", "ARM64Features":
			// The CPU feature queries. Spec describes operations, not the
			// hardware features that implement them.
			return true
		}
	case "simd":
		switch {
		case recv == "" && (name == "VectorBits" || name == "Emulated"):
			// Queries about how package simd is implemented on the running
			// hardware, which have no counterpart in spec's model of
			// vector operations.
			return true
		case recv != "" && name == "ToArch":
			// The conversions between the portable types and archsimd's
			// types, which spec doesn't describe because it has no notion
			// of the architecture-specific types (SPEC-TRANSITION.md §2.7).
			return portableTypes[recv]
		case recv == "" && strings.HasSuffix(name, "FromArch"):
			// The other direction of those conversions, such as
			// Int8sFromArch.
			return portableTypes[strings.TrimSuffix(name, "FromArch")]
		}
	}
	return false
}

// portableTypes are package simd's portable vector and mask types.
var portableTypes = map[string]bool{
	"Int8s": true, "Int16s": true, "Int32s": true, "Int64s": true,
	"Uint8s": true, "Uint16s": true, "Uint32s": true, "Uint64s": true,
	"Float32s": true, "Float64s": true,
	"Mask8s": true, "Mask16s": true, "Mask32s": true, "Mask64s": true,
}
