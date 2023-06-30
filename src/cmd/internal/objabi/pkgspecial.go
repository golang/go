// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import "sync"

// PkgSpecial indicates special build properties of a given runtime-related
// package.
type PkgSpecial struct {
	// AllowAsmABI indicates that assembly in this package is allowed to use ABI
	// selectors in symbol names. Generally this is needed for packages that
	// interact closely with the runtime package or have performance-critical
	// assembly.
	AllowAsmABI bool
}

var allowAsmABIPkgs = []string{
	"runtime",
	"reflect",
	"syscall",
	"internal/bytealg",
	"runtime/internal/syscall",
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
		for _, pkg := range allowAsmABIPkgs {
			set(pkg, func(ps *PkgSpecial) { ps.AllowAsmABI = true })
		}
	})
	return pkgSpecials[pkgPath]
}
