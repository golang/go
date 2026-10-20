// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"internal/pkgbits"
	"os"

	"cmd/compile/internal/base"
	"cmd/compile/internal/importer"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
)

// This file holds the loGo/s facts-sidecar reader (M3-S1). Upstream files
// carry only hook lines; all logic lives here.
//
// readLogosFacts scans the archive for the __.LOGOS member following
// __.PKGDEF (re-seeking from the start, so readExportData keeps its
// signature) and adopts its bytes into the importer overlay registry.
// It reports whether the member was found: archives without it predate
// the sidecar (plain-Go packages) and import as before, and the Go
// union-import check runs only when it is present, so those scopes are
// never resolved. A Go importer (go/importer) reads only __.PKGDEF and
// never sees it.
func readLogosFacts(f *os.File, path, data string) (bool, error) {
	member, err := facts.FindMember(f)
	if err != nil {
		return false, err
	}
	if member == nil {
		return false, nil
	}
	_, err = facts.Import(path, data, member)
	return true, err
}

// logosImporters records the importers serving loGo/s packages. It is
// keyed by importer because gcimports is an upstream type that takes
// hook lines only. A missing entry means a Go importer: it rejects a
// dual-use union at the import (M2-S7), while a loGo/s importer keeps
// the full union.
var logosImporters = make(map[*gcimports]bool)

// logosMarkImporter records whether m imports for a loGo/s package. It
// sets the entry unconditionally so a recycled importer address never
// inherits a stale kind across typecheck rechecks.
func logosMarkImporter(m *gcimports, isLogos bool) {
	logosImporters[m] = isLogos
}

// logosImportFrom imports path for m, selecting the package kind from
// the importer mark: only the loGo/s kind keeps dual-use unions legal.
func logosImportFrom(m *gcimports, path string) (*types2.Package, error) {
	_, pkg, err := readImportFileImpl(path, typecheck.Target, m.ctxt, m.packages, logosImporters[m])
	return pkg, err
}

func readImportFileImpl(path string, target *ir.Package, env *types2.Context, packages map[string]*types2.Package, isLogos bool) (pkg1 *types.Pkg, pkg2 *types2.Package, err error) {
	path, err = resolveImportPath(path)
	if err != nil {
		return
	}

	if path == "unsafe" {
		pkg1, pkg2 = types.UnsafePkg, types2.Unsafe

		// TODO(mdempsky): Investigate if this actually matters. Why would
		// the linker or runtime care whether a package imported unsafe?
		if !pkg1.Direct {
			pkg1.Direct = true
			target.Imports = append(target.Imports, pkg1)
		}

		return
	}

	pkg1 = types.NewPkg(path, "")
	if packages != nil {
		pkg2 = packages[path]
		if !(pkg1.Direct == (pkg2 != nil && pkg2.Complete())) {

			base.Fatalf("pkg1.Direct == (pkg2 != nil && pkg2.Complete()), path=%s\npackages=%v", path, packages)
		}
	}

	if pkg1.Direct {
		return
	}
	pkg1.Direct = true
	target.Imports = append(target.Imports, pkg1)

	f, err := openPackage(path)
	if err != nil {
		return
	}
	defer f.Close()

	data, err := readExportData(f)
	if err != nil {
		return
	}
	var logosFacts bool
	if logosFacts, err = readLogosFacts(f, path, data); err != nil {
		return
	}

	if base.Debug.Export != 0 {
		fmt.Printf("importing %s (%s)\n", path, f.Name())
	}

	pr := pkgbits.NewPkgDecoder(pkg1.Path, data)

	// The union-import check needs the package scope, but resolving a
	// foreign lazy scope tripped importer asserts. Gate it on the
	// export naming a loGo/s source file, which plain-Go archives
	// never do; the sidecar check additionally requires the transport
	// to be intact.
	checkLogosUnion := !isLogos && logosFacts && logosExportHasLogosFile(&pr)

	// Read package descriptors for both types2 and compiler backend.
	readPackage(newPkgReader(pr), pkg1, false)
	pkg2 = importer.ReadPackage(env, packages, pr)
	if pkg2 != nil && isLogos {
		pkg2 = applyLogosOverlayForLgoImporter(path, pkg2, packages) // logos M3-S7 hook: .lgo importers see Option types.
	}
	if pkg2 != nil && checkLogosUnion {
		checkLogosGoImport(pkg2)
	}

	err = addFingerprint(path, data)
	return
}
