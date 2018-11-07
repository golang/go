// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"go/format"
)

// Format formats a document with a given range.
func Format(ctx context.Context, f *File, rng Range) ([]TextEdit, error) {
	fAST, err := f.GetAST()
	if err != nil {
		return nil, err
	}

	// TODO(rstambler): use astutil.PathEnclosingInterval to
	// find the largest ast.Node n contained within start:end, and format the
	// region n.Pos-n.End instead.

	// format.Node changes slightly from one release to another, so the version
	// of Go used to build the LSP server will determine how it formats code.
	// This should be acceptable for all users, who likely be prompted to rebuild
	// the LSP server on each Go release.
	buf := &bytes.Buffer{}
	if err := format.Node(buf, f.view.Config.Fset, fAST); err != nil {
		return nil, err
	}
	// TODO(rstambler): Compute text edits instead of replacing whole file.
	return []TextEdit{
		{
			Range:   rng,
			NewText: buf.String(),
		},
	}, nil
}
