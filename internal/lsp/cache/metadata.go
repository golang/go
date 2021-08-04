// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"go/types"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/span"
)

// Declare explicit types for package paths, names, and IDs to ensure that we
// never use an ID where a path belongs, and vice versa. If we confused these,
// it would result in confusing errors because package IDs often look like
// package paths.
type (
	PackageID   string
	PackagePath string
	PackageName string
)

// Metadata holds package Metadata extracted from a call to packages.Load.
type Metadata struct {
	ID              PackageID
	PkgPath         PackagePath
	Name            PackageName
	GoFiles         []span.URI
	CompiledGoFiles []span.URI
	ForTest         PackagePath
	TypesSizes      types.Sizes
	Errors          []packages.Error
	Deps            []PackageID
	MissingDeps     map[PackagePath]struct{}
	Module          *packages.Module
	depsErrors      []*packagesinternal.PackageError

	// Config is the *packages.Config associated with the loaded package.
	Config *packages.Config

	// IsIntermediateTestVariant reports whether the given package is an
	// intermediate test variant, e.g.
	// "golang.org/x/tools/internal/lsp/cache [golang.org/x/tools/internal/lsp/source.test]".
	IsIntermediateTestVariant bool
}

// Name implements the source.Metadata interface.
func (m *Metadata) PackageName() string {
	return string(m.Name)
}

// PkgPath implements the source.Metadata interface.
func (m *Metadata) PackagePath() string {
	return string(m.PkgPath)
}

// KnownMetadata is a wrapper around metadata that tracks its validity.
type KnownMetadata struct {
	*Metadata

	// Valid is true if the given metadata is Valid.
	// Invalid metadata can still be used if a metadata reload fails.
	Valid bool

	// ShouldLoad is true if the given metadata should be reloaded.
	ShouldLoad bool
}
