// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mod provides core features related to go.mod file
// handling for use by Go editors and tools.
package mod

import (
	"context"
	"fmt"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/trace"
)

func Diagnostics(ctx context.Context, snapshot source.Snapshot) (source.FileIdentity, []source.Diagnostic, error) {
	// TODO: We will want to support diagnostics for go.mod files even when the -modfile flag is turned off.
	realfh, tempfh, err := snapshot.ModFiles(ctx)
	if err != nil {
		return source.FileIdentity{}, nil, err
	}
	// Check the case when the tempModfile flag is turned off.
	if realfh == nil || tempfh == nil {
		return source.FileIdentity{}, nil, nil
	}

	ctx, done := trace.StartSpan(ctx, "modfiles.Diagnostics", telemetry.File.Of(realfh.Identity().URI))
	defer done()

	// If the view has a temporary go.mod file, we want to run "go mod tidy" to be able to
	// diff between the real and the temp files.
	cfg := snapshot.View().Config(ctx)
	args := append([]string{"mod", "tidy"}, cfg.BuildFlags...)
	if _, err := source.InvokeGo(ctx, snapshot.View().Folder().Filename(), cfg.Env, args...); err != nil {
		return source.FileIdentity{}, nil, err
	}

	realMod, err := snapshot.View().Session().Cache().ParseModHandle(realfh).Parse(ctx)
	if err != nil {
		return source.FileIdentity{}, nil, err
	}
	tempMod, err := snapshot.View().Session().Cache().ParseModHandle(tempfh).Parse(ctx)
	if err != nil {
		return source.FileIdentity{}, nil, err
	}

	reports := []source.Diagnostic{}
	// Check indirect vs direct, and removal of dependencies.
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
		diag := &source.Diagnostic{
			Message:  fmt.Sprintf("%s is not used in this module.", dep),
			Source:   "go mod tidy",
			Range:    protocol.Range{Start: getPos(req.Syntax.Start), End: getPos(req.Syntax.End)},
			Severity: protocol.SeverityWarning,
		}
		if tempReqs[dep] != nil && req.Indirect != tempReqs[dep].Indirect {
			diag.Message = fmt.Sprintf("%s should be an indirect dependency.", dep)
			if req.Indirect {
				diag.Message = fmt.Sprintf("%s should not be an indirect dependency.", dep)
			}
		}
		reports = append(reports, *diag)
	}
	return realfh.Identity(), reports, nil
}

// TODO: Check to see if we need to go through internal/span.
func getPos(pos modfile.Position) protocol.Position {
	return protocol.Position{
		Line:      float64(pos.Line - 1),
		Character: float64(pos.LineRune - 1),
	}
}
