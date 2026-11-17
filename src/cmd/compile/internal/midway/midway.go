// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/types2"
	"fmt"
	"internal/buildcfg"
	"slices"
	"strconv"
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

// SimdMark is added to names of types/functions/variables that have been
// specialized to a particular SIMD vector width, and is followed by the
// width of the vector in bits, or "0" if the specialization is for emulated
// SIMD.
const SimdMark = "@simd"

// SimdInitMark prefixes synthetic package-level proxy variables created by
// midway for top-level variable initializers that depend on SIMD even though
// the variable's type does not (including blank "_" initializers).
//
// The proxy name has the form:
//
//	"@simdinit<vecSize>@<target>"
//
// where <target> is either the original package variable's name (e.g., "N")
// or "_@<id>" for a blank initializer. Because the name starts with "@", it is
// never exported and cannot collide with legal Go identifiers. The unified IR
// writer omits these proxy declarations from pkgDecl and the unified IR reader
// rewrites their initializers to conditionally assign directly to the original
// variable (or blank).
const SimdInitMark = SimdMark + "init"

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
	return strings.Contains(name, SimdMark)
}

// SimdInitProxyName returns the synthetic init-proxy variable name for target
// at the given SIMD vector bit size (0 for emulation).
func SimdInitProxyName(target string, vecSize int) string {
	return fmt.Sprintf("%s%d@%s", SimdInitMark, vecSize, target)
}

// IsSimdInitProxy reports whether symName is a synthetic SIMD init-proxy
// variable name created by midway.
func IsSimdInitProxy(symName string) bool {
	return strings.HasPrefix(symName, SimdInitMark)
}

// ParseSimdInitProxy parses a synthetic SIMD init-proxy name of the form
// "@simdinit<vecSize>@<target>". If <target> encodes a blank
// initializer ("_" or "_@<id>"), baseName is returned as "_".
func ParseSimdInitProxy(symName string) (baseName string, vecSize int, ok bool) {
	if !strings.HasPrefix(symName, SimdInitMark) {
		return "", 0, false
	}
	rest := symName[len(SimdInitMark):]
	at := strings.IndexByte(rest, '@')
	if at <= 0 || at+1 >= len(rest) {
		return "", 0, false
	}
	vecSize, ok = parseVecSize(rest[:at])
	if !ok {
		return "", 0, false
	}
	target := rest[at+1:]
	if target == "_" || strings.HasPrefix(target, "_@") {
		return "_", vecSize, true
	}
	return target, vecSize, true
}

// ParseSimdSuffix splits a midway-specialized symbol name of the form
// "<baseName>@simd<vecSize>" (for example, "x@simd0", "x@simd256")
// into its original base name and numeric vector bit size (0 for emulation).
// It returns ok == false if symName does not have a valid "@simd<digits>" suffix.
func ParseSimdSuffix(symName string) (baseName string, vecSize int, ok bool) {
	if IsSimdInitProxy(symName) {
		return "", 0, false
	}
	idx := strings.LastIndex(symName, SimdMark)
	if idx < 0 {
		return "", 0, false
	}
	vecSize, ok = parseVecSize(symName[idx+len(SimdMark):])
	if !ok {
		return "", 0, false
	}
	return symName[:idx], vecSize, true
}

func parseVecSize(s string) (vecSize int, ok bool) {
	if len(s) == 0 {
		return 0, false
	}
	for i := range len(s) {
		if s[i] < '0' || s[i] > '9' {
			return 0, false
		}
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, false
	}
	return n, true
}

// isPreSimdInit reports whether init is the original (pre-rewrite) package-level
// variable initializer for a declaration that midway specialized into "@simd..."
// variables or "@simdinit..." conditional init proxies. Midway retains the
// original var declaration so unrewritten references continue to typecheck and
// export data stays intact, and always generates a size-0 (emulated) counterpart
// ("<name>@simd0" or "@simdinit0@<name>"); the original initializer must be
// dropped so it does not execute at runtime alongside the guarded specialized
// initializers.
func isPreSimdInit(init *types2.Initializer) bool {
	for _, v := range init.Lhs {
		name := v.Name()
		if name == "_" || IsSimdInitProxy(name) || strings.Contains(name, SimdMark) || v.Pkg() == nil {
			continue
		}
		scope := v.Pkg().Scope()
		if scope.Lookup(name+SimdMark+"0") != nil || scope.Lookup(SimdInitProxyName(name, 0)) != nil {
			return true
		}
	}
	return false
}

// DropPreSimdInits removes pre-rewrite dummy initializers from info.InitOrder
// after types2 re-checks the rewritten AST.
func DropPreSimdInits(info *types2.Info) {
	info.InitOrder = slices.DeleteFunc(info.InitOrder, isPreSimdInit)
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
