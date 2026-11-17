// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmp"
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

// InitOrderPlan records the pre-midway Go-spec package variable initialization
// order so it can be restored after midway specializes top-level variable
// declarations and types2 re-checks the rewritten AST.
//
// Splitting a package variable into per-size declarations allows types2 to
// interleave specializations of one variable with unrelated declarations during
// re-check (for example, when a non-SIMD var depends only on the last proxy),
// which can violate Go-spec side-effect ordering under non-default SIMD configs.
type InitOrderPlan struct {
	numSlots   int
	slotByName map[string]int     // records order of named initialized variables
	slotByPos  map[syntax.Pos]int // records order of "_" targets of initializations, identified by their position.
}

func saveInitOrder(info *types2.Info) *InitOrderPlan {
	p := &InitOrderPlan{
		numSlots:   len(info.InitOrder),
		slotByName: make(map[string]int, len(info.InitOrder)),
		slotByPos:  make(map[syntax.Pos]int),
	}
	for i, init := range info.InitOrder {
		hasNamed := false
		for _, v := range init.Lhs {
			if v.Name() != "_" {
				p.slotByName[v.Name()] = i
				hasNamed = true
			}
		}
		if !hasNamed && len(init.Lhs) > 0 {
			p.slotByPos[init.Lhs[0].Pos()] = i
		}
	}
	return p
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

// Apply filters out pre-rewrite dummy initializers from info.InitOrder and
// reorders all remaining initializers so that every declaration's specialized
// initializers execute atomically at the exact Go-spec position of its original
// declaration.
func (p *InitOrderPlan) Apply(info *types2.Info) {
	if p == nil {
		return
	}

	type slotEntry struct {
		init    *types2.Initializer
		vecSize int
	}
	slots := make([][]slotEntry, p.numSlots)
	var extras []*types2.Initializer

	for _, init := range info.InitOrder {
		if isPreSimdInit(init) {
			continue
		}
		slot := -1
		vecSize := -1
		blankOnly := true
		for _, v := range init.Lhs {
			name := v.Name()
			baseName := name
			if bn, sz, ok := ParseSimdInitProxy(name); ok {
				baseName = bn
				vecSize = sz
			} else if bn, sz, ok := ParseSimdSuffix(name); ok {
				baseName = bn
				vecSize = sz
			}
			// Figure out the slot (index in the init order) from
			// the first named variable on the LHS.
			if slot < 0 && baseName != "_" {
				if s, ok := p.slotByName[baseName]; ok {
					slot = s
					blankOnly = false
				}
			}
		}
		// If there were no named variables,
		// use the first "_" variable's position to determine the slot.
		if slot < 0 && len(init.Lhs) > 0 {
			if s, ok := p.slotByPos[init.Lhs[0].Pos()]; ok {
				slot = s
			}
		}
		if slot >= 0 {
			slots[slot] = append(slots[slot], slotEntry{init: init, vecSize: vecSize})
		} else {
			// Any variable without a slot was introduced as part of a midway rewrite.
			// These should have only package-import-correctness side-effects, not
			// data flow side effects.
			if !blankOnly {
				base.Fatalf("Non-blank initialized variable added by midway rewriting")
			}
			extras = append(extras, init)
		}
	}

	// each element of slots is either
	//   a singleton, for a variable that was not rewritten for simd,
	//   or a slice of simd-size-specialized variables,
	//   or a slice of simd-size-specialized conditional-initialization proxies.
	//
	// Each slice must be sorted for determinism, and all the initializations
	// must be appended into a new order consistent that by construction will be
	// consistent with the original order.

	result := make([]*types2.Initializer, 0, len(info.InitOrder))
	for _, entries := range slots {
		if len(entries) > 1 {
			slices.SortFunc(entries, func(a, b slotEntry) int {
				return cmp.Compare(a.vecSize, b.vecSize)
			})
		}
		for _, e := range entries {
			result = append(result, e.init)
		}
	}
	result = append(result, extras...)
	info.InitOrder = result
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
