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
	use, pathStart, pathEnd := usePath(pw, hoverRng.Start)

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

func usePath(pw *source.ParsedWorkFile, pos token.Pos) (use *modfile.Use, pathStart, pathEnd int) {
	for _, u := range pw.File.Use {
		path := []byte(u.Path)
		s, e := u.Syntax.Start.Byte, u.Syntax.End.Byte
		i := bytes.Index(pw.Mapper.Content[s:e], path)
		if i == -1 {
			// This should not happen.
			continue
		}
		// Shift the start position to the location of the
		// module directory within the use statement.
		pathStart, pathEnd = s+i, s+i+len(path)
		if token.Pos(pathStart) <= pos && pos <= token.Pos(pathEnd) {
			return u, pathStart, pathEnd
		}
	}
	return nil, 0, 0
}
