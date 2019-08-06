// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/token"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	errors "golang.org/x/xerrors"
)

type tokenKey struct {
	file source.FileIdentity
}

type tokenHandle struct {
	handle *memoize.Handle
	file   source.FileHandle
}

type tokenData struct {
	memoize.NoCopy

	tok *token.File
	err error
}

func (c *cache) TokenHandle(fh source.FileHandle) source.TokenHandle {
	key := tokenKey{
		file: fh.Identity(),
	}
	h := c.store.Bind(key, func(ctx context.Context) interface{} {
		data := &tokenData{}
		data.tok, data.err = tokenFile(ctx, c, fh)
		return data
	})
	return &tokenHandle{
		handle: h,
		file:   fh,
	}
}

func (h *tokenHandle) File() source.FileHandle {
	return h.file
}

func (h *tokenHandle) Token(ctx context.Context) (*token.File, error) {
	v := h.handle.Get(ctx)
	if v == nil {
		return nil, ctx.Err()
	}
	data := v.(*tokenData)
	return data.tok, data.err
}

func tokenFile(ctx context.Context, c *cache, fh source.FileHandle) (*token.File, error) {
	// First, check if we already have a parsed AST for this file's handle.
	for _, mode := range []source.ParseMode{
		source.ParseHeader,
		source.ParseExported,
		source.ParseFull,
	} {
		pk := parseKey{
			file: fh.Identity(),
			mode: mode,
		}
		pd, ok := c.store.Cached(pk).(*parseGoData)
		if !ok {
			continue
		}
		if pd.ast == nil {
			continue
		}
		if !pd.ast.Pos().IsValid() {
			continue
		}
		return c.FileSet().File(pd.ast.Pos()), nil
	}
	// We have not yet parsed this file.
	buf, _, err := fh.Read(ctx)
	if err != nil {
		return nil, err
	}
	tok := c.FileSet().AddFile(fh.Identity().URI.Filename(), -1, len(buf))
	if tok == nil {
		return nil, errors.Errorf("no token.File for %s", fh.Identity().URI)
	}
	tok.SetLinesForContent(buf)
	return tok, nil
}
