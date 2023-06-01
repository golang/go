// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/token"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/event"
)

// TypeDefinition handles the textDocument/typeDefinition request for Go files.
func TypeDefinition(ctx context.Context, snapshot Snapshot, fh FileHandle, position protocol.Position) ([]protocol.Location, error) {
	ctx, done := event.Start(ctx, "source.TypeDefinition")
	defer done()

	pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, fh.URI())
	if err != nil {
		return nil, err
	}
	pos, err := pgf.PositionPos(position)
	if err != nil {
		return nil, err
	}

	// TODO(rfindley): handle type switch implicits correctly here: if the user
	// jumps to the type definition of x in x := y.(type), it makes sense to jump
	// to the type of y.
	_, obj, _ := referencedObject(pkg, pgf, pos)
	if obj == nil {
		return nil, nil
	}

	tname := typeToObject(obj.Type())
	if tname == nil {
		return nil, fmt.Errorf("no type definition for %s", obj.Name())
	}

	if !tname.Pos().IsValid() {
		// The only defined types with no position are error and comparable.
		if tname.Name() != "error" && tname.Name() != "comparable" {
			bug.Reportf("unexpected type name with no position: %s", tname)
		}
		return nil, nil
	}

	loc, err := mapPosition(ctx, pkg.FileSet(), snapshot, tname.Pos(), tname.Pos()+token.Pos(len(tname.Name())))
	if err != nil {
		return nil, err
	}
	return []protocol.Location{loc}, nil
}
