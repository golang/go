// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"sort"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func (s *Server) codeLens(ctx context.Context, params *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	snapshot, fh, ok, release, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	defer release()
	if !ok {
		return nil, err
	}
	var lensFuncs map[string]source.LensFunc
	switch fh.Kind() {
	case source.Mod:
		lensFuncs = mod.LensFuncs()
	case source.Go:
		lensFuncs = source.LensFuncs()
	default:
		// Unsupported file kind for a code lens.
		return nil, nil
	}
	var result []protocol.CodeLens
	for lens, lf := range lensFuncs {
		if !snapshot.View().Options().Codelenses[lens] {
			continue
		}
		added, err := lf(ctx, snapshot, fh)
		// Code lens is called on every keystroke, so we should just operate in
		// a best-effort mode, ignoring errors.
		if err != nil {
			event.Error(ctx, fmt.Sprintf("code lens %s failed", lens), err)
			continue
		}
		result = append(result, added...)
	}
	sort.Slice(result, func(i, j int) bool {
		a, b := result[i], result[j]
		if protocol.CompareRange(a.Range, b.Range) == 0 {
			return a.Command.Command < b.Command.Command
		}
		return protocol.CompareRange(a.Range, b.Range) < 0
	})
	return result, nil
}
