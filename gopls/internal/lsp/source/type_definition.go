// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/token"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/event"
)

// TypeDefinition handles the textDocument/typeDefinition request for Go files.
func TypeDefinition(ctx context.Context, snapshot Snapshot, fh FileHandle, position protocol.Position) ([]protocol.Location, error) {
	ctx, done := event.Start(ctx, "source.TypeDefinition")
	defer done()

	pkg, pgf, err := PackageForFile(ctx, snapshot, fh.URI(), TypecheckFull, NarrowestPackage)
	if err != nil {
		return nil, err
	}
	pos, err := pgf.PositionPos(position)
	if err != nil {
		return nil, err
	}

	obj := referencedObject(pkg, pgf, pos)
	if obj == nil {
		return nil, nil
	}

	typObj := typeToObject(obj.Type())
	if typObj == nil {
		return nil, fmt.Errorf("no type definition for %s", obj.Name())
	}

	// Identifiers with the type "error" are a special case with no position.
	if hasErrorType(typObj) {
		// TODO(rfindley): we can do better here, returning a link to the builtin
		// file.
		return nil, nil
	}

	loc, err := mapPosition(ctx, pkg.FileSet(), snapshot, typObj.Pos(), typObj.Pos()+token.Pos(len(typObj.Name())))
	if err != nil {
		return nil, err
	}
	return []protocol.Location{loc}, nil
}
