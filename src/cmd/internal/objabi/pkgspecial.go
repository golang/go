// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import "sync"

// PkgSpecial indicates special build properties of a given runtime-related
// package.
type PkgSpecial struct {
	// Runtime indicates that this package is "runtime" or imported by
	// "runtime". This has several effects (which maybe should be split out):
	//
	// - Implicit allocation is disallowed.
	//
	// - Various runtime pragmas are enabled.
	//
	// - Optimizations are always enabled.
	//
	// This should be set for runtime and all packages it imports, and may be
	// set for additional packages.
	Runtime bool

	// NoInstrument indicates this package should not receive sanitizer
	// instrumentation. In many of these, instrumentation could cause infinite
	// recursion. This is all runtime packages, plus those that support the
	// sanitizers.
	NoInstrument bool

	// NoRaceFunc indicates functions in this package should not get
	// racefuncenter/racefuncexit instrumentation Memory accesses in these
	// packages are either uninteresting or will cause false positives.
	NoRaceFunc bool

	// AllowAsmABI indicates that assembly in this package is allowed to use ABI
	// selectors in symbol names. Generally this is needed for packages that
	// interact closely with the runtime package or have performance-critical
	// assembly.
	AllowAsmABI bool
}

var runtimePkgs = []string{
	"runtime",

	"internal/runtime/atomic",
	"runtime/internal/math",
	"runtime/internal/sys",
	"internal/runtime/syscall",

	"internal/abi",
	"internal/bytealg",
	"internal/byteorder",
	"internal/chacha8rand",
	"internal/coverage/rtcov",
	"internal/cpu",
	"internal/goarch",
	"internal/godebugs",
	"internal/goexperiment",
	"internal/goos",
	"internal/profilerecord",
	"internal/stringslite",
}

// extraNoInstrumentPkgs is the set of packages in addition to runtimePkgs that
// should have NoInstrument set.
var extraNoInstrumentPkgs = []string{
	"runtime/race",
	"runtime/msan",
	"runtime/asan",
	// We omit bytealg even though it's imported by runtime because it also
	// backs a lot of package bytes. Currently we don't have a way to omit race
	// instrumentation when used from the runtime while keeping race
	// instrumentation when used from user code. Somehow this doesn't seem to
	// cause problems, though we may be skating on thin ice. See #61204.
	"-internal/bytealg",
}

var noRaceFuncPkgs = []string{"sync", "sync/atomic", "internal/runtime/atomic"}

var allowAsmABIPkgs = []string{
	"runtime",
	"reflect",
	"syscall",
	"internal/bytealg",
	"internal/chacha8rand",
	"internal/runtime/syscall",
	"runtime/internal/startlinetest",
}

var (
	pkgSpecials     map[string]PkgSpecial
	pkgSpecialsOnce sync.Once
)

// LookupPkgSpecial returns special build properties for the given package path.
func LookupPkgSpecial(pkgPath string) PkgSpecial {
	pkgSpecialsOnce.Do(func() {
		// Construct pkgSpecials from various package lists. This lets us use
		// more flexible logic, while keeping the final map simple, and avoids
		// the init-time cost of a map.
		pkgSpecials = make(map[string]PkgSpecial)
		set := func(elt string, f func(*PkgSpecial)) {
			s := pkgSpecials[elt]
			f(&s)
			pkgSpecials[elt] = s
		}
		for _, pkg := range runtimePkgs {
			set(pkg, func(ps *PkgSpecial) { ps.Runtime = true; ps.NoInstrument = true })
		}
		for _, pkg := range extraNoInstrumentPkgs {
			if pkg[0] == '-' {
				set(pkg[1:], func(ps *PkgSpecial) { ps.NoInstrument = false })
			} else {
				set(pkg, func(ps *PkgSpecial) { ps.NoInstrument = true })
			}
		}
		for _, pkg := range noRaceFuncPkgs {
			set(pkg, func(ps *PkgSpecial) { ps.NoRaceFunc = true })
		}
		for _, pkg := range allowAsmABIPkgs {
			set(pkg, func(ps *PkgSpecial) { ps.AllowAsmABI = true })
		}
	})
	return pkgSpecials[pkgPath]
}
