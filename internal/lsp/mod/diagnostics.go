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

		rng, err := getRangeFromPositions(m, realfh, req.Syntax.Start, req.Syntax.End)
		if err != nil {
			return nil, err
		}
		var diag *source.Diagnostic

		if tempReqs[dep] != nil && req.Indirect != tempReqs[dep].Indirect {
			// Show diagnostics for dependencies that are incorrectly labeled indirect.
			if req.Indirect {
				var fix []source.SuggestedFix
				// If the dependency should not be indirect, just highlight the // indirect.
				if comments := req.Syntax.Comment(); comments != nil && len(comments.Suffix) > 0 {
					end := comments.Suffix[0].Start
					end.LineRune += len(comments.Suffix[0].Token)
					end.Byte += len([]byte(comments.Suffix[0].Token))

					rng, err = getRangeFromPositions(m, realfh, comments.Suffix[0].Start, end)
					if err != nil {
						return nil, err
					}
					fix = []source.SuggestedFix{
						{
							Title: "Remove indirect",
							Edits: map[span.URI][]protocol.TextEdit{realfh.Identity().URI: []protocol.TextEdit{
								{
									Range:   rng,
									NewText: "",
								},
							}},
						},
					}
				}
				diag = &source.Diagnostic{
					Message:        fmt.Sprintf("%s should be a direct dependency.", dep),
					Range:          rng,
					SuggestedFixes: fix,
					Source:         "go mod tidy",
					Severity:       protocol.SeverityWarning,
				}
			} else {
				diag = &source.Diagnostic{
					Message:  fmt.Sprintf("%s should be an indirect dependency.", dep),
					Range:    rng,
					Source:   "go mod tidy",
					Severity: protocol.SeverityWarning,
				}
			}
		}
		// Handle unused dependencies.
		if tempReqs[dep] == nil {
			diag = &source.Diagnostic{
				Message: fmt.Sprintf("%s is not used in this module.", dep),
				Range:   rng,
				SuggestedFixes: []source.SuggestedFix{
					{
						Title: fmt.Sprintf("Remove %s.", dep),
						Edits: map[span.URI][]protocol.TextEdit{realfh.Identity().URI: []protocol.TextEdit{
							{
								Range:   rng,
								NewText: "",
							},
						}},
					},
				},
				Source:   "go mod tidy",
				Severity: protocol.SeverityWarning,
			}
		}
		reports[realfh.Identity()] = append(reports[realfh.Identity()], *diag)
	}
	return reports, nil
}

// TODO: Add caching for go.mod diagnostics to be able to map them back to source.Diagnostics
// and reuse the cached suggested fixes.
func SuggestedFixes(fh source.FileHandle, diags []protocol.Diagnostic) []protocol.CodeAction {
	var actions []protocol.CodeAction
	for _, diag := range diags {
		var title string
		if strings.Contains(diag.Message, "is not used in this module") {
			split := strings.Split(diag.Message, " ")
			if len(split) < 1 {
				continue
			}
			title = fmt.Sprintf("Remove dependency: %s", split[0])
		}
		if strings.Contains(diag.Message, "should be a direct dependency.") {
			title = "Remove indirect"
		}
		if title == "" {
			continue
		}
		actions = append(actions, protocol.CodeAction{
			Title: title,
			Kind:  protocol.QuickFix,
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: []protocol.TextDocumentEdit{
					{
						TextDocument: protocol.VersionedTextDocumentIdentifier{
							Version: fh.Identity().Version,
							TextDocumentIdentifier: protocol.TextDocumentIdentifier{
								URI: protocol.NewURI(fh.Identity().URI),
							},
						},
						Edits: []protocol.TextEdit{protocol.TextEdit{Range: diag.Range, NewText: ""}},
					},
				},
			},
			Diagnostics: diags,
		})
	}
	return actions
}

func getEndOfLine(req *modfile.Require, m *protocol.ColumnMapper) (span.Point, error) {
	comments := req.Syntax.Comment()
	if comments == nil {
		return positionToPoint(m, req.Syntax.End)
	}
	suffix := comments.Suffix
	if len(suffix) == 0 {
		return positionToPoint(m, req.Syntax.End)
	}
	end := suffix[0].Start
	end.LineRune += len(suffix[0].Token)
	return positionToPoint(m, end)
}

func getRangeFromPositions(m *protocol.ColumnMapper, fh source.FileHandle, s, e modfile.Position) (protocol.Range, error) {
	start, err := positionToPoint(m, s)
	if err != nil {
		return protocol.Range{}, err
	}
	end, err := positionToPoint(m, e)
	if err != nil {
		return protocol.Range{}, err
	}
	spn := span.New(fh.Identity().URI, start, end)
	rng, err := m.Range(spn)
	if err != nil {
		return protocol.Range{}, err
	}
	return rng, nil
}

func positionToPoint(m *protocol.ColumnMapper, pos modfile.Position) (span.Point, error) {
	line, col, err := m.Converter.ToPosition(pos.Byte)
	if err != nil {
		return span.Point{}, err
	}
	return span.NewPoint(line, col, pos.Byte), nil
}
