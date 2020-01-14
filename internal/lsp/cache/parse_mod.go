// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
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
	mapper  *protocol.ColumnMapper
	err     error
}

func (c *cache) ParseModHandle(fh source.FileHandle) source.ParseModHandle {
	h := c.store.Bind(fh.Identity(), func(ctx context.Context) interface{} {
		data := &parseModData{}
		data.modfile, data.mapper, data.err = parseMod(ctx, fh)
		return data
	})
	return &parseModHandle{
		handle: h,
		file:   fh,
	}
}

func parseMod(ctx context.Context, fh source.FileHandle) (*modfile.File, *protocol.ColumnMapper, error) {
	ctx, done := trace.StartSpan(ctx, "cache.parseMod", telemetry.File.Of(fh.Identity().URI.Filename()))
	defer done()

	buf, _, err := fh.Read(ctx)
	if err != nil {
		return nil, nil, err
	}
	parsed, err := modfile.Parse(fh.Identity().URI.Filename(), buf, nil)
	if err != nil {
		// TODO(golang/go#36486): This can be removed when modfile.Parse returns structured errors.
		re := regexp.MustCompile(`.*:([\d]+): (.+)`)
		matches := re.FindStringSubmatch(strings.TrimSpace(err.Error()))
		if len(matches) < 3 {
			log.Error(ctx, "could not parse golang/x/mod error message", err)
			return nil, nil, err
		}
		line, e := strconv.Atoi(matches[1])
		if e != nil {
			return nil, nil, err
		}
		contents := strings.Split(string(buf), "\n")[line-1]
		return nil, nil, &source.Error{
			Message: matches[2],
			Range: protocol.Range{
				Start: protocol.Position{Line: float64(line - 1), Character: float64(0)},
				End:   protocol.Position{Line: float64(line - 1), Character: float64(len(contents))},
			},
		}
	}
	m := &protocol.ColumnMapper{
		URI:       fh.Identity().URI,
		Converter: span.NewContentConverter(fh.Identity().URI.Filename(), buf),
		Content:   buf,
	}
	return parsed, m, nil
}

func (pgh *parseModHandle) String() string {
	return pgh.File().Identity().URI.Filename()
}

func (pgh *parseModHandle) File() source.FileHandle {
	return pgh.file
}

func (pgh *parseModHandle) Parse(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, error) {
	v := pgh.handle.Get(ctx)
	if v == nil {
		return nil, nil, errors.Errorf("no parsed file for %s", pgh.File().Identity().URI)
	}
	data := v.(*parseModData)
	return data.modfile, data.mapper, data.err
}
