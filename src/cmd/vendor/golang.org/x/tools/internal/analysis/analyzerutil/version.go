// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analyzerutil

import (
	"go/ast"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/packagepath"
	"golang.org/x/tools/internal/stdlib"
	"golang.org/x/tools/internal/versions"
)

// FileUsesGoVersion reports whether the specified file may use features of the
// specified version of Go (e.g. "go1.24").
//
// Tip: we recommend using this check "late", just before calling
// pass.Report, rather than "early" (when entering each ast.File, or
// each candidate node of interest, during the traversal), because the
// operation is not free, yet is not a highly selective filter: the
// fraction of files that pass most version checks is high and
// increases over time.
func FileUsesGoVersion(pass *analysis.Pass, file *ast.File, version string) (_res bool) {
	fileVersion := pass.TypesInfo.FileVersions[file]

	// Standard packages that are part of toolchain bootstrapping
	// are not considered to use a version of Go later than the
	// current bootstrap toolchain version.
	// The bootstrap rule does not cover tests,
	// and some tests (e.g. debug/elf/file_test.go) rely on this.
	pkgpath := pass.Pkg.Path()
	if packagepath.IsStdPackage(pkgpath) &&
		stdlib.IsBootstrapPackage(pkgpath) && // (excludes "*_test" external test packages)
		!strings.HasSuffix(pass.Fset.File(file.Pos()).Name(), "_test.go") { // (excludes all tests)
		fileVersion = stdlib.BootstrapVersion.String() // package must bootstrap
	}

	return !versions.Before(fileVersion, version)
}
