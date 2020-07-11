// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"

	"golang.org/x/tools/internal/lsp/analysis/fillstruct"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

func FillStruct(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) ([]protocol.TextDocumentEdit, error) {
	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, NarrowestPackageHandle)
	if err != nil {
		return nil, fmt.Errorf("getting file for Identifier: %w", err)
	}
	file, _, m, _, err := pgh.Cached()
	if err != nil {
		return nil, err
	}
	spn, err := m.RangeSpan(pRng)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	content, err := fh.Read()
	if err != nil {
		return nil, err
	}
	fset := snapshot.View().Session().Cache().FileSet()
	fix, err := fillstruct.SuggestedFix(fset, rng.Start, content, file, pkg.GetTypes(), pkg.GetTypesInfo())
	if err != nil {
		return nil, err
	}
	var edits []protocol.TextDocumentEdit
	for _, edit := range fix.TextEdits {
		rng := span.NewRange(fset, edit.Pos, edit.End)
		spn, err = rng.Span()
		if err != nil {
			return nil, nil
		}
		clRng, err := m.Range(spn)
		if err != nil {
			return nil, nil
		}
		edits = append(edits, protocol.TextDocumentEdit{
			TextDocument: protocol.VersionedTextDocumentIdentifier{
				Version: fh.Version(),
				TextDocumentIdentifier: protocol.TextDocumentIdentifier{
					URI: protocol.URIFromSpanURI(fh.URI()),
				},
			},
			Edits: []protocol.TextEdit{
				{
					Range:   clRng,
					NewText: string(edit.NewText),
				},
			},
		})
	}
	return edits, nil
}
