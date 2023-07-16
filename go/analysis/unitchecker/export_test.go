// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unitchecker

import (
	"go/token"
	"go/types"
)

// This file exposes various internal hooks to the separate_test.
//
// TODO(adonovan): expose a public API to unitchecker that doesn't
// rely on details of JSON .cfg files or enshrine I/O decisions or
// assumptions about how "go vet" locates things. Ideally the new Run
// function would accept an interface, and a Config file would be just
// one way--the go vet way--to implement it.

func SetTypeImportExport(
	MakeTypesImporter func(*Config, *token.FileSet) types.Importer,
	ExportTypes func(*Config, *token.FileSet, *types.Package) error,
) {
	makeTypesImporter = MakeTypesImporter
	exportTypes = ExportTypes
}
