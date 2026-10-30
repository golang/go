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

// FileGoVersion returns the effective Go version of the specified
// file (e.g. "go1.24"), and reports whether it is known.
//
// The version is unknown when the type checker did not record a valid
// version for the file, such as for parsed files that are ignored by
// the type checker, or when neither the file nor the package
// specifies a version, or in application that has not been updated to
// populate the [types.Config.GoVersion] field added in Go 1.18.
//
// For standard packages that are part of toolchain bootstrapping,
// the result is the bootstrap toolchain version.
//
// Most analyzers should use the simpler [FileUsesGoVersion].
// Use this function when you need to distinguish "unknown" from
// "before", for example, to enable a check only for files that are
// known to use an older version of Go.
func FileGoVersion(pass *analysis.Pass, file *ast.File) (version string, known bool) {
	fileVersion := pass.TypesInfo.FileVersions[file]
	if fileVersion == "" || !versions.IsValid(fileVersion) {
		return "", false // e.g. IgnoredFiles, or no Config.GoVersion
	}

	// Standard packages that are part of toolchain bootstrapping
	// are not considered to use a version of Go later than the
	// current bootstrap toolchain version.
	// The bootstrap rule does not cover tests,
	// and some tests (e.g. debug/elf/file_test.go) rely on this.
	pkgpath := pass.Pkg.Path()
	if packagepath.MaybeStdPackage(pkgpath) &&
		stdlib.IsBootstrapPackage(pkgpath) && // (excludes "*_test" external test packages)
		!strings.HasSuffix(pass.Fset.File(file.Pos()).Name(), "_test.go") { // (excludes all tests)
		fileVersion = stdlib.BootstrapVersion.String() // package must bootstrap
	}

	return fileVersion, true
}

// FileUsesGoVersion reports whether the specified file may use
// features of the specified version of Go (e.g. "go1.24").
//
// It returns false when version information is not available,
// such as for parsed files that are ignored by the type checker.
// Use [FileGoVersion] to distinguish "unknown" from "before".
//
// Tip: we recommend using this check "late", just before calling
// pass.Report, rather than "early" (when entering each ast.File, or
// each candidate node of interest, during the traversal), because the
// operation is not free, yet is not a highly selective filter: the
// fraction of files that pass most version checks is high and
// increases over time.
func FileUsesGoVersion(pass *analysis.Pass, file *ast.File, version string) bool {
	fileVersion, known := FileGoVersion(pass, file)
	return known && !versions.Before(fileVersion, version)
}
