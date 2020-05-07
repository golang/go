// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/token"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
)

// CodeLens computes code lens for Go source code.
func CodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	if !snapshot.View().Options().EnabledCodeLens[CommandGenerate] {
		return nil, nil
	}
	f, _, m, _, err := snapshot.View().Session().Cache().ParseGoHandle(fh, ParseFull).Parse(ctx)
	if err != nil {
		return nil, err
	}
	const ggDirective = "//go:generate"
	for _, c := range f.Comments {
		for _, l := range c.List {
			if !strings.HasPrefix(l.Text, ggDirective) {
				continue
			}
			fset := snapshot.View().Session().Cache().FileSet()
			rng, err := newMappedRange(fset, m, l.Pos(), l.Pos()+token.Pos(len(ggDirective))).Range()
			if err != nil {
				return nil, err
			}
			dir := filepath.Dir(fh.Identity().URI.Filename())
			return []protocol.CodeLens{
				{
					Range: rng,
					Command: protocol.Command{
						Title:     "run go generate",
						Command:   CommandGenerate,
						Arguments: []interface{}{dir, false},
					},
				},
				{
					Range: rng,
					Command: protocol.Command{
						Title:     "run go generate ./...",
						Command:   CommandGenerate,
						Arguments: []interface{}{dir, true},
					},
				},
			}, nil

		}
	}
	return nil, nil
}
