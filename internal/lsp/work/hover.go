// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bytes"
	"context"
	"go/token"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	errors "golang.org/x/xerrors"
)

func Hover(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, position protocol.Position) (*protocol.Hover, error) {
	// We only provide hover information for the view's go.work file.
	if fh.URI() != snapshot.WorkFile() {
		return nil, nil
	}

	ctx, done := event.Start(ctx, "work.Hover")
	defer done()

	// Get the position of the cursor.
	pw, err := snapshot.ParseWork(ctx, fh)
	if err != nil {
		return nil, errors.Errorf("getting go.work file handle: %w", err)
	}
	spn, err := pw.Mapper.PointSpan(position)
	if err != nil {
		return nil, errors.Errorf("computing cursor position: %w", err)
	}
	hoverRng, err := spn.Range(pw.Mapper.Converter)
	if err != nil {
		return nil, errors.Errorf("computing hover range: %w", err)
	}

	// Confirm that the cursor is inside a use statement, and then find
	// the position of the use statement's directory path.
	var use *modfile.Use
	var pathStart, pathEnd int
	for _, u := range pw.File.Use {
		dep := []byte(u.Path)
		s, e := u.Syntax.Start.Byte, u.Syntax.End.Byte
		i := bytes.Index(pw.Mapper.Content[s:e], dep)
		if i == -1 {
			// This should not happen.
			continue
		}
		// Shift the start position to the location of the
		// module directory within the use statement.
		pathStart, pathEnd = s+i, s+i+len(dep)
		if token.Pos(pathStart) <= hoverRng.Start && hoverRng.Start <= token.Pos(pathEnd) {
			use = u
			break
		}
	}

	// The cursor position is not on a use statement.
	if use == nil {
		return nil, nil
	}

	// Get the mod file denoted by the use.
	modfh, err := snapshot.GetFile(ctx, modFileURI(pw, use))
	pm, err := snapshot.ParseMod(ctx, modfh)
	if err != nil {
		return nil, errors.Errorf("getting modfile handle: %w", err)
	}
	mod := pm.File.Module.Mod

	// Get the range to highlight for the hover.
	rng, err := source.ByteOffsetsToRange(pw.Mapper, fh.URI(), pathStart, pathEnd)
	if err != nil {
		return nil, err
	}
	options := snapshot.View().Options()
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  options.PreferredContentFormat,
			Value: mod.Path,
		},
		Range: rng,
	}, nil
}
