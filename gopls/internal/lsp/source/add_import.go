// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"

	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// AddImport adds a single import statement to the given file
func AddImport(ctx context.Context, snapshot Snapshot, fh VersionedFileHandle, importPath string) ([]protocol.TextEdit, error) {
	_, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return nil, err
	}
	return ComputeOneImportFixEdits(snapshot, pgf, &imports.ImportFix{
		StmtInfo: imports.ImportInfo{
			ImportPath: importPath,
		},
		FixType: imports.AddImport,
	})
}
