// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

type parseModHandle struct {
	handle *memoize.Handle
	file   source.FileHandle
}

type parseModData struct {
	memoize.NoCopy

	modfile *modfile.File
	err     error
}

func (c *cache) ParseModHandle(fh source.FileHandle) source.ParseModHandle {
	key := parseKey{
		file: fh.Identity(),
		mode: source.ParseFull,
	}
	h := c.store.Bind(key, func(ctx context.Context) interface{} {
		data := &parseModData{}
		data.modfile, data.err = parseMod(ctx, fh)
		return data
	})
	return &parseModHandle{
		handle: h,
		file:   fh,
	}
}

func parseMod(ctx context.Context, fh source.FileHandle) (modifle *modfile.File, err error) {
	ctx, done := trace.StartSpan(ctx, "cache.parseMod", telemetry.File.Of(fh.Identity().URI.Filename()))
	defer done()

	buf, _, err := fh.Read(ctx)
	if err != nil {
		return nil, err
	}
	f, err := modfile.Parse(fh.Identity().URI.Filename(), buf, nil)
	if err != nil {
		return nil, err
	}
	return f, nil
}

func (pgh *parseModHandle) String() string {
	return pgh.File().Identity().URI.Filename()
}

func (pgh *parseModHandle) File() source.FileHandle {
	return pgh.file
}

func (pgh *parseModHandle) Parse(ctx context.Context) (*modfile.File, error) {
	v := pgh.handle.Get(ctx)
	if v == nil {
		return nil, errors.Errorf("no parsed file for %s", pgh.File().Identity().URI)
	}
	data := v.(*parseModData)
	return data.modfile, data.err
}
