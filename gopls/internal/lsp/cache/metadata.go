// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"go/types"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/packagesinternal"
)

type (
	PackageID   = source.PackageID
	PackagePath = source.PackagePath
	PackageName = source.PackageName
	ImportPath  = source.ImportPath
)

// Metadata holds package Metadata extracted from a call to packages.Load.
type Metadata struct {
	ID              PackageID
	PkgPath         PackagePath
	Name            PackageName
	GoFiles         []span.URI
	CompiledGoFiles []span.URI
	ForTest         PackagePath // package path under test, or ""
	TypesSizes      types.Sizes
	Errors          []packages.Error
	DepsByImpPath   map[ImportPath]PackageID  // may contain dups; empty ID => missing
	DepsByPkgPath   map[PackagePath]PackageID // values are unique and non-empty
	Module          *packages.Module
	depsErrors      []*packagesinternal.PackageError

	// Config is the *packages.Config associated with the loaded package.
	Config *packages.Config
}

// PackageID implements the source.Metadata interface.
func (m *Metadata) PackageID() PackageID { return m.ID }

// Name implements the source.Metadata interface.
func (m *Metadata) PackageName() PackageName { return m.Name }

// PkgPath implements the source.Metadata interface.
func (m *Metadata) PackagePath() PackagePath { return m.PkgPath }

// IsIntermediateTestVariant reports whether the given package is an
// intermediate test variant, e.g. "net/http [net/url.test]".
//
// Such test variants arise when an x_test package (in this case net/url_test)
// imports a package (in this case net/http) that itself imports the the
// non-x_test package (in this case net/url).
//
// This is done so that the forward transitive closure of net/url_test has
// only one package for the "net/url" import.
// The intermediate test variant exists to hold the test variant import:
//
// net/url_test [net/url.test]
//
//	| "net/http" -> net/http [net/url.test]
//	| "net/url" -> net/url [net/url.test]
//	| ...
//
// net/http [net/url.test]
//
//	| "net/url" -> net/url [net/url.test]
//	| ...
//
// This restriction propagates throughout the import graph of net/http: for
// every package imported by net/http that imports net/url, there must be an
// intermediate test variant that instead imports "net/url [net/url.test]".
//
// As one can see from the example of net/url and net/http, intermediate test
// variants can result in many additional packages that are essentially (but
// not quite) identical. For this reason, we filter these variants wherever
// possible.
func (m *Metadata) IsIntermediateTestVariant() bool {
	return m.ForTest != "" && m.ForTest != m.PkgPath && m.ForTest+"_test" != m.PkgPath
}

// ModuleInfo implements the source.Metadata interface.
func (m *Metadata) ModuleInfo() *packages.Module {
	return m.Module
}

// KnownMetadata is a wrapper around metadata that tracks its validity.
type KnownMetadata struct {
	*Metadata

	// Valid is true if the given metadata is Valid.
	// Invalid metadata can still be used if a metadata reload fails.
	Valid bool
}
