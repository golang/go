// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
)

// ArchInfo contains all architecture-specific naming conventions.
type ArchInfo struct {
	Arch      string // e.g., "amd64", "arm64"
	ArchUpper string // SSA op prefix, e.g., "AMD64", "ARM64". SVE shares "ARM64".
	ObjArch   string // e.g., "x86", "arm64" (for cmd/internal/obj/*)
	// GoTypeArch identifies a distinct simdgen target and is the single tag that
	// names its generated output: the Go API files (types_<GoTypeArch>.go,
	// ops_<GoTypeArch>.go) and, in the backend package, simdssa_<GoTypeArch>.go.
	// It equals Arch for amd64/arm64, but is "sve" for the SVE target, so SVE's
	// output sits alongside — not on top of — the NEON arm64 files even though
	// both compile as GOARCH arm64.
	//
	// It is also the key of the shared simdgenericOps.go merge: each target's
	// run tags its generic ops with // ARCH:<GoTypeArch> so a later run for a
	// different target unions its ops in without dropping the others. Keying on
	// Arch instead would make an SVE run strip the "arm64" tag off every NEON
	// generic op (and drop arm64-only ones), since SVE and NEON share Arch.
	//
	// TODO: once the NEON and SVE type/op sets are unified, arm64 and sve can
	// collapse back into a single target and this second tag can go away.
	GoTypeArch string
	// SIMDTag names this target's generated files and functions in the
	// architecture-agnostic backend directories: ssa/_gen/simd<TAG>ops.go and
	// simd<TAG>.rules, and ssagen/simd<TAG>intrinsics.go, along with the
	// simd<TAG>Ops and simd<TAG>Intrinsics functions they define.
	SIMDTag         string
	RegInfoKeys     []string        // RegInfo shapes that generate SSA lowering code (gen_simdssa.go)
	RegInfoSet      map[string]bool // Valid regInfo shapes (for gen_simdMachineOps.go)
	RegInfoParams   string          // Function parameter declaration for generated ops (simd[AMD64|ARM64]ops.go)
	GeneratedHeader string          // Header comment for generated files
	// Scalable reports that this target's vectors have a length that is only
	// known at run time, so the generated types carry no fixed lane count. It is
	// a property of the instruction set, independent of whether the target
	// shares a backend package: a future RVV target is scalable and owns its
	// package, LSX/LASX are neither.
	Scalable     bool
	Arrangements []string // SIMD arrangement suffixes (e.g., "4S", "2D" for ARM64; nil for amd64)
}

var amd64RegInfoKeys = []string{
	"v11",
	"v21",
	"v2k",
	"v2kv",
	"v2kk",
	"vkv",
	"v31",
	"v3kv",
	"v11Imm8",
	"vkvImm8",
	"v21Imm8",
	"v2kImm8",
	"v2kkImm8",
	"v31ResultInArg0",
	"v3kvResultInArg0",
	"vfpv",
	"vfpkv",
	"vgpvImm8",
	"vgpvImm",
	"vgpImm8",
	"vgpImm",
	"v2kvImm8",
	"vkvload",
	"v21load",
	"v31loadResultInArg0",
	"v3kvloadResultInArg0",
	"v2kvload",
	"v2kload",
	"v11load",
	"v11loadImm8",
	"vkvloadImm8",
	"v21loadImm8",
	"v2kloadImm8",
	"v2kkloadImm8",
	"v2kvloadImm8",
	"v31ResultInArg0Imm8",
	"v31loadResultInArg0Imm8",
	"v21ResultInArg0",
	"v21ResultInArg0Imm8",
	"v31x0AtIn2ResultInArg0",
	"v2kvResultInArg0",
}

var arm64RegInfoKeys = []string{
	"v11",
	"v11Imm",
	"v11ImmIn1",
	"v11Scalar",
	"v11ScalarImmIn1",
	"v21",
	"v21Imm",
	"v31ResultInArg0",
	"vgpImmIn1",
	"vgpvResultInArg0ImmOutIn0",
	"vfpvResultInArg0ImmOutIn1",
	"v11Long",
	"v11Narrow",
	"v11ImmNarrow",
	"v11ImmLong",
	"v21Long",
	"v11Long2",
	"v21Narrow2",
	"v21ImmNarrow2",
	"v11ImmLong2",
	"v21Long2",
	"v21List",
	"v31ResultInArg0List",
}

var amd64RegInfoSet = map[string]bool{
	"v11": true, "v21": true, "v2k": true, "v2kv": true, "v2kk": true, "vkv": true, "v31": true, "v3kv": true, "vgpv": true, "vgp": true, "vfpv": true, "vfpkv": true,
	"w11": true, "w21": true, "w2k": true, "w2kw": true, "w2kk": true, "wkw": true, "w31": true, "w3kw": true, "wgpw": true, "wgp": true, "wfpw": true, "wfpkw": true,
	"wkwload": true, "v21load": true, "v31load": true, "v11load": true, "w21load": true, "w31load": true, "w2kload": true, "w2kwload": true, "w11load": true,
	"w3kwload": true, "w2kkload": true, "v31x0AtIn2": true,
}

var arm64RegInfoSet = map[string]bool{
	"v11":                 true,
	"v21":                 true,
	"v21Imm":              true,
	"v31":                 true,
	"vgp":                 true,
	"vgpv":                true,
	"vfpv":                true,
	"v11ImmIn1":           true,
	"v11Long":             true,
	"v11Narrow":           true,
	"v11ImmNarrow":        true,
	"v11ImmLong":          true,
	"v21Long":             true,
	"v11Long2":            true,
	"v21Narrow2":          true,
	"v21ImmNarrow2":       true,
	"v11ImmLong2":         true,
	"v21Long2":            true,
	"v21List":             true,
	"v31ResultInArg0List": true,
}

// arm64Arrangements contains the SIMD arrangement suffixes for ARM64 NEON.
// These correspond to the ARNG_* constants in cmd/internal/obj/arm64/a.out.go.
var arm64Arrangements = []string{
	"8B", "16B", "1D", "4H", "8H", "2S", "4S", "2D", "1Q", "B", "H", "S", "D",
}

// sveArrangements contains the per-element arrangement letters for ARM64 SVE.
// SVE vectors are scalable, so the arrangement only encodes the element width.
var sveArrangements = []string{"B", "H", "S", "D"}

// SVE regInfo shapes. SVE scalable-vector (Z) registers use a "z" letter to keep
// their names — and thus the generated simd<Shape> lowering helpers — distinct
// from the NEON "v" shapes. Predicate (P) registers will add "p" shapes as
// predicated ops are supported. The names are the parameters of the generated
// simdARM64SVEOps function, bound to concrete regInfo values in ARM64Ops.go.
var sveRegInfoKeys = []string{
	"z11",      // 1 Z in, 1 Z out (unary, e.g. NEG)
	"z21",      // 2 Z in, 1 Z out (binary, e.g. unpredicated ADD)
	"z2kk",     // 2 Z in, 1 P (governing predicate) in, 1 P out (predicated compare, e.g. ZCMPGT)
	"z2kv",     // 2 Z in, 1 P (select predicate) in, 1 Z out (constructive, e.g. ZSEL)
	"z2kvPred", // 2 Z in, 1 P (governing predicate) in, 1 Z out (destructive, e.g. ZADD/M)
	// 3 Z in, 1 P (governing predicate) in, 1 Z out, destination shared with the
	// first input: a destructive predicated op behind a MOVPRFX, e.g. ZADDMergingPrefixed.
	"z3kvPredResultInArg0",
}

var sveRegInfoSet = map[string]bool{
	"z11":      true,
	"z21":      true,
	"z2kk":     true,
	"z2kv":     true,
	"z2kvPred": true,
	"z3kvPred": true,
}

const sveRegInfoParams = "z11, z21, z2kk, z2kv, z2kvPred, z3kvPred regInfo"

const sveGeneratedHeader = `// Code generated by 'simdgen -o godefs -goroot $GOROOT -arch sve -arm64Path $ARM64_ISA_PATH go_sve.yaml types.yaml categories.yaml'; DO NOT EDIT.
`

const amd64RegInfoParams = "v11, v21, v2k, vkv, v2kv, v2kk, v31, v3kv, vgpv, vgp, vfpv, vfpkv, w11, w21, w2k, wkw, w2kw, w2kk, w31, w3kw, wgpw, wgp, wfpw, wfpkw,\n\twkwload, v21load, v31load, v11load, w21load, w31load, w2kload, w2kwload, w11load, w3kwload, w2kkload, v31x0AtIn2 regInfo"

const arm64RegInfoParams = "v11, v21, v31, vgp, vgpv, vfpv regInfo"

const amd64GeneratedHeader = `// Code generated by 'simdgen -o godefs -goroot $GOROOT -arch amd64 -xedPath $XED_PATH go_amd64.yaml types.yaml categories.yaml'; DO NOT EDIT.
`

const arm64GeneratedHeader = `// Code generated by 'simdgen -o godefs -goroot $GOROOT -arch arm64 -arm64Path $ARM64_ISA_PATH go_arm64.yaml types.yaml categories.yaml'; DO NOT EDIT.
`

// isSVE reports whether this target is ARM64 SVE.
func (a ArchInfo) isSVE() bool { return a.GoTypeArch == "sve" }

// sharesBackendPackage reports whether this target's generated ssa-lowering
// code lands in a backend package another target already owns, so its file and
// function names must not collide with that target's. It is about the output
// layout, not about any property of the instruction set.
func (a ArchInfo) sharesBackendPackage() bool { return a.GoTypeArch != a.Arch }

// ssaGenFile returns the basename of the generated ssa-to-prog lowering file in
// internal/<Arch>/. The primary target keeps the historical "simdssa.go"; a
// one sharing another's uses a tagged name so it sits beside it.
func (a ArchInfo) ssaGenFile() string {
	if a.sharesBackendPackage() {
		return "simdssa_" + a.GoTypeArch + ".go"
	}
	return "simdssa.go"
}

// ssaGenFuncInfix returns the infix for the generated ssaGenSIMD<infix>Value
// function so a target sharing another's backend package does not collide with
// it. Empty when the target owns its package.
func (a ArchInfo) ssaGenFuncInfix() string {
	if a.sharesBackendPackage() {
		return strings.ToUpper(a.GoTypeArch)
	}
	return ""
}

// GetArchInfo returns architecture-specific information based on the target architecture.
func GetArchInfo(arch string) (ArchInfo, error) {
	switch arch {
	case "amd64":
		return ArchInfo{
			Arch:            "amd64",
			ArchUpper:       "AMD64",
			ObjArch:         "x86",
			GoTypeArch:      "amd64",
			SIMDTag:         "AMD64",
			RegInfoKeys:     amd64RegInfoKeys,
			RegInfoSet:      amd64RegInfoSet,
			RegInfoParams:   amd64RegInfoParams,
			GeneratedHeader: amd64GeneratedHeader,
		}, nil
	case "arm64":
		return ArchInfo{
			Arch:            "arm64",
			ArchUpper:       "ARM64",
			ObjArch:         "arm64",
			GoTypeArch:      "arm64",
			SIMDTag:         "ARM64",
			RegInfoKeys:     arm64RegInfoKeys,
			RegInfoSet:      arm64RegInfoSet,
			RegInfoParams:   arm64RegInfoParams,
			GeneratedHeader: arm64GeneratedHeader,
			Arrangements:    arm64Arrangements,
		}, nil
	case "sve":
		// SVE targets arm64 (shared OpARM64 SSA prefix and obj/arm64 assembler),
		// but its generated files use the "SVE"/"sve" tags so they sit alongside
		// the NEON arm64 files instead of clobbering them. SVE registers are the
		// scalable Z (vectors) and P (predicates) banks, with their own regInfo
		// shapes (z11, z21, ...) and hand-written ssaGenValue helpers.
		return ArchInfo{
			Arch:            "arm64",
			ArchUpper:       "ARM64",
			ObjArch:         "arm64",
			GoTypeArch:      "sve",
			SIMDTag:         "ARM64SVE",
			RegInfoKeys:     sveRegInfoKeys,
			RegInfoSet:      sveRegInfoSet,
			RegInfoParams:   sveRegInfoParams,
			GeneratedHeader: sveGeneratedHeader,
			Arrangements:    sveArrangements,
			Scalable:        true,
		}, nil
	default:
		return ArchInfo{}, fmt.Errorf("unsupported architecture: %s", arch)
	}
}

// CurrentArch returns the ArchInfo for the current FlagArch setting.
func CurrentArch() ArchInfo {
	info, err := GetArchInfo(*FlagArch)
	if err != nil {
		panic(err)
	}
	return info
}
