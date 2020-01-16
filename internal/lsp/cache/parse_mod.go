// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

const ModTidyError = "go mod tidy"
const SyntaxError = "syntax"

type parseModKey struct {
	snapshot source.Snapshot
	cfg      string
}

type parseModHandle struct {
	handle *memoize.Handle
	file   source.FileHandle
	cfg    *packages.Config
}

type parseModData struct {
	memoize.NoCopy

	modfile     *modfile.File
	mapper      *protocol.ColumnMapper
	parseErrors []source.Error
	err         error
}

func (pgh *parseModHandle) String() string {
	return pgh.File().Identity().URI.Filename()
}

func (pgh *parseModHandle) File() source.FileHandle {
	return pgh.file
}

func (pgh *parseModHandle) Parse(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, []source.Error, error) {
	v := pgh.handle.Get(ctx)
	if v == nil {
		return nil, nil, nil, errors.Errorf("no parsed file for %s", pgh.File().Identity().URI)
	}
	data := v.(*parseModData)
	return data.modfile, data.mapper, data.parseErrors, data.err
}

func (s *snapshot) ParseModHandle(ctx context.Context, fh source.FileHandle) source.ParseModHandle {
	realfh, tempfh, err := s.ModFiles(ctx)
	cfg := s.View().Config(ctx)
	folder := s.View().Folder().Filename()

	key := parseModKey{
		snapshot: s,
		cfg:      hashConfig(cfg),
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		data := &parseModData{}
		if err != nil {
			data.err = err
			return data
		}
		// Check the case when the tempModfile flag is turned off.
		if realfh == nil || tempfh == nil {
			return data
		}
		data.modfile, data.mapper, data.parseErrors, data.err = goModFileDiagnostics(ctx, realfh, tempfh, cfg, folder)
		return data
	})
	return &parseModHandle{
		handle: h,
		file:   fh,
		cfg:    cfg,
	}
}

func goModFileDiagnostics(ctx context.Context, realfh, tempfh source.FileHandle, cfg *packages.Config, folder string) (*modfile.File, *protocol.ColumnMapper, []source.Error, error) {
	ctx, done := trace.StartSpan(ctx, "cache.parseMod", telemetry.File.Of(realfh.Identity().URI.Filename()))
	defer done()

	// Copy the real go.mod file content into the temp go.mod file.
	contents, err := ioutil.ReadFile(realfh.Identity().URI.Filename())
	if err != nil {
		return nil, nil, nil, err
	}
	if err := ioutil.WriteFile(tempfh.Identity().URI.Filename(), contents, os.ModePerm); err != nil {
		return nil, nil, nil, err
	}

	// We want to run "go mod tidy" to be able to diff between the real and the temp files.
	args := append([]string{"mod", "tidy"}, cfg.BuildFlags...)
	if _, err := source.InvokeGo(ctx, folder, cfg.Env, args...); err != nil {
		// Ignore parse errors here. They'll be handled below.
		if !strings.Contains(err.Error(), "errors parsing go.mod") {
			return nil, nil, nil, err
		}
	}

	realMod, m, parseErr, err := parseModFile(ctx, realfh)
	if parseErr != nil {
		return nil, nil, []source.Error{*parseErr}, nil
	}
	if err != nil {
		return nil, nil, nil, err
	}

	tempMod, _, _, err := parseModFile(ctx, tempfh)
	if err != nil {
		return nil, nil, nil, err
	}

	errors, err := modRequireErrors(realfh, m, realMod, tempMod)
	if err != nil {
		return nil, nil, nil, err
	}
	return realMod, m, errors, nil
}

func modParseErrors(ctx context.Context, uri span.URI, m *protocol.ColumnMapper, modTidyErr error, buf []byte) (source.Error, error) {
	re := regexp.MustCompile(`.*:([\d]+): (.+)`)
	matches := re.FindStringSubmatch(strings.TrimSpace(modTidyErr.Error()))
	if len(matches) < 3 {
		log.Error(ctx, "could not parse golang/x/mod error message", modTidyErr)
		return source.Error{}, modTidyErr
	}
	line, err := strconv.Atoi(matches[1])
	if err != nil {
		return source.Error{}, modTidyErr
	}
	lines := strings.Split(string(buf), "\n")
	if len(lines) <= line {
		return source.Error{}, errors.Errorf("could not parse goland/x/mod error message, line number out of range")
	}
	// Get the length of the line that the error is present on.
	endOfLine := len(lines[line-1])
	sOffset, err := m.Converter.ToOffset(line, 0)
	if err != nil {
		return source.Error{}, err
	}
	eOffset, err := m.Converter.ToOffset(line, endOfLine)
	if err != nil {
		return source.Error{}, err
	}
	spn := span.New(uri, span.NewPoint(line, 0, sOffset), span.NewPoint(line, endOfLine, eOffset))
	rng, err := m.Range(spn)
	if err != nil {
		return source.Error{}, err
	}
	return source.Error{
		Category: SyntaxError,
		Message:  matches[2],
		Range:    rng,
		URI:      uri,
	}, nil
}

func modRequireErrors(realfh source.FileHandle, m *protocol.ColumnMapper, realMod, tempMod *modfile.File) ([]source.Error, error) {
	realReqs := make(map[string]*modfile.Require, len(realMod.Require))
	tempReqs := make(map[string]*modfile.Require, len(tempMod.Require))
	for _, req := range realMod.Require {
		realReqs[req.Mod.Path] = req
	}
	for _, req := range tempMod.Require {
		realReq := realReqs[req.Mod.Path]
		if realReq != nil && realReq.Indirect == req.Indirect {
			delete(realReqs, req.Mod.Path)
		}
		tempReqs[req.Mod.Path] = req
	}

	var errors []source.Error
	for _, req := range realReqs {
		if req.Syntax == nil {
			continue
		}
		dep := req.Mod.Path
		// Handle dependencies that are incorrectly labeled indirect and vice versa.
		if tempReqs[dep] != nil && req.Indirect != tempReqs[dep].Indirect {
			directErr, err := modDirectnessErrors(realfh, m, req)
			if err != nil {
				return nil, err
			}
			errors = append(errors, directErr)
		}
		// Handle unused dependencies.
		if tempReqs[dep] == nil {
			rng, err := rangeFromPositions(realfh.Identity().URI, m, req.Syntax.Start, req.Syntax.End)
			if err != nil {
				return nil, err
			}
			errors = append(errors, source.Error{
				Category: ModTidyError,
				Message:  fmt.Sprintf("%s is not used in this module.", dep),
				Range:    rng,
				URI:      realfh.Identity().URI,
			})
		}
	}
	return errors, nil
}

func modDirectnessErrors(fh source.FileHandle, m *protocol.ColumnMapper, req *modfile.Require) (source.Error, error) {
	rng, err := rangeFromPositions(fh.Identity().URI, m, req.Syntax.Start, req.Syntax.End)
	if err != nil {
		return source.Error{}, err
	}
	if req.Indirect {
		// If the dependency should be direct, just highlight the // indirect.
		if comments := req.Syntax.Comment(); comments != nil && len(comments.Suffix) > 0 {
			end := comments.Suffix[0].Start
			end.LineRune += len(comments.Suffix[0].Token)
			end.Byte += len([]byte(comments.Suffix[0].Token))
			rng, err = rangeFromPositions(fh.Identity().URI, m, comments.Suffix[0].Start, end)
			if err != nil {
				return source.Error{}, err
			}
		}
		return source.Error{
			Category: ModTidyError,
			Message:  fmt.Sprintf("%s should be a direct dependency.", req.Mod.Path),
			Range:    rng,
			URI:      fh.Identity().URI,
		}, nil
	}
	return source.Error{
		Category: ModTidyError,
		Message:  fmt.Sprintf("%s should be an indirect dependency.", req.Mod.Path),
		Range:    rng,
		URI:      fh.Identity().URI,
	}, nil
}

func parseModFile(ctx context.Context, fh source.FileHandle) (*modfile.File, *protocol.ColumnMapper, *source.Error, error) {
	contents, _, err := fh.Read(ctx)
	if err != nil {
		return nil, nil, nil, err
	}
	m := &protocol.ColumnMapper{
		URI:       fh.Identity().URI,
		Converter: span.NewContentConverter(fh.Identity().URI.Filename(), contents),
		Content:   contents,
	}
	parsed, err := modfile.Parse(fh.Identity().URI.Filename(), contents, nil)
	if err != nil {
		parseErr, err := modParseErrors(ctx, fh.Identity().URI, m, err, contents)
		return nil, nil, &parseErr, err
	}
	return parsed, m, nil, nil
}

func rangeFromPositions(uri span.URI, m *protocol.ColumnMapper, s, e modfile.Position) (protocol.Range, error) {
	line, col, err := m.Converter.ToPosition(s.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	start := span.NewPoint(line, col, s.Byte)

	line, col, err = m.Converter.ToPosition(e.Byte)
	if err != nil {
		return protocol.Range{}, err
	}
	end := span.NewPoint(line, col, e.Byte)

	spn := span.New(uri, start, end)
	rng, err := m.Range(spn)
	if err != nil {
		return protocol.Range{}, err
	}
	return rng, nil
}
