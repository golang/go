// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/trace"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (map[source.FileIdentity][]source.Diagnostic, error) {
	// TODO: We will want to support diagnostics for go.mod files even when the -modfile flag is turned off.
	realfh, tempfh, err := snapshot.ModFiles(ctx)
	if err != nil {
		return nil, err
	}
	// Check the case when the tempModfile flag is turned off.
	if realfh == nil || tempfh == nil {
		return nil, nil
	}

	ctx, done := trace.StartSpan(ctx, "modfiles.Diagnostics", telemetry.File.Of(realfh.Identity().URI))
	defer done()

	// If the view has a temporary go.mod file, we want to run "go mod tidy" to be able to
	// diff between the real and the temp files.
	cfg := snapshot.View().Config(ctx)
	args := append([]string{"mod", "tidy"}, cfg.BuildFlags...)
	if _, err := source.InvokeGo(ctx, snapshot.View().Folder().Filename(), cfg.Env, args...); err != nil {
		// Ignore parse errors here. They'll be handled below.
		if !strings.Contains(err.Error(), "errors parsing go.mod") {
			return nil, err
		}
	}

	realMod, m, err := snapshot.View().Session().Cache().ParseModHandle(realfh).Parse(ctx)
	// If the go.mod file fails to parse, return errors right away.
	if err, ok := err.(*source.Error); ok {
		return map[source.FileIdentity][]source.Diagnostic{
			realfh.Identity(): []source.Diagnostic{{
				Message:  err.Message,
				Source:   "syntax",
				Range:    err.Range,
				Severity: protocol.SeverityError,
			}},
		}, nil
	}
	if err != nil {
		return nil, err
	}
	tempMod, _, err := snapshot.View().Session().Cache().ParseModHandle(tempfh).Parse(ctx)
	if err != nil {
		return nil, err
	}

	// Check indirect vs direct, and removal of dependencies.
	reports := map[source.FileIdentity][]source.Diagnostic{
		realfh.Identity(): []source.Diagnostic{},
	}
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
	for _, req := range realReqs {
		if req.Syntax == nil {
			continue
		}
		dep := req.Mod.Path

		start, err := positionToPoint(m, req.Syntax.Start)
		if err != nil {
			return nil, err
		}
		end, err := positionToPoint(m, req.Syntax.End)
		if err != nil {
			return nil, err
		}
		spn := span.New(realfh.Identity().URI, start, end)
		rng, err := m.Range(spn)
		if err != nil {
			return nil, err
		}

		diag := &source.Diagnostic{
			Message:  fmt.Sprintf("%s is not used in this module.", dep),
			Source:   "go mod tidy",
			Range:    rng,
			Severity: protocol.SeverityWarning,
		}
		if tempReqs[dep] != nil && req.Indirect != tempReqs[dep].Indirect {
			diag.Message = fmt.Sprintf("%s should be an indirect dependency.", dep)
			if req.Indirect {
				diag.Message = fmt.Sprintf("%s should not be an indirect dependency.", dep)
			}
		}
		reports[realfh.Identity()] = append(reports[realfh.Identity()], *diag)
	}
	return reports, nil
}

func positionToPoint(m *protocol.ColumnMapper, pos modfile.Position) (span.Point, error) {
	line, col, err := m.Converter.ToPosition(pos.Byte)
	if err != nil {
		return span.Point{}, err
	}
	return span.NewPoint(line, col, pos.Byte), nil
}
