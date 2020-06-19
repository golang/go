// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"io/ioutil"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/span"
)

type modTidyKey struct {
	sessionID       string
	cfg             string
	gomod           string
	imports         string
	unsavedOverlays string
	view            string
}

type modTidyHandle struct {
	handle *memoize.Handle

	pmh source.ParseModHandle
}

type modTidyData struct {
	memoize.NoCopy

	// missingDeps contains dependencies that should be added to the view's
	// go.mod file.
	missingDeps map[string]*modfile.Require

	// diagnostics are any errors and associated suggested fixes for
	// the go.mod file.
	diagnostics []source.Error

	err error
}

func (mth *modTidyHandle) Tidy(ctx context.Context) (map[string]*modfile.Require, []source.Error, error) {
	v := mth.handle.Get(ctx)
	if v == nil {
		return nil, nil, ctx.Err()
	}
	data := v.(*modTidyData)
	return data.missingDeps, data.diagnostics, data.err
}

func (s *snapshot) ModTidyHandle(ctx context.Context) (source.ModTidyHandle, error) {
	if !s.view.tmpMod {
		return nil, source.ErrTmpModfileUnsupported
	}
	if handle := s.getModTidyHandle(); handle != nil {
		return handle, nil
	}
	fh, err := s.GetFile(ctx, s.view.modURI)
	if err != nil {
		return nil, err
	}
	pmh, err := s.ParseModHandle(ctx, fh)
	if err != nil {
		return nil, err
	}
	wsPackages, err := s.WorkspacePackages(ctx)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if err != nil {
		return nil, err
	}
	imports, err := hashImports(ctx, wsPackages)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	overlayHash := hashUnsavedOverlays(s.files)
	s.mu.Unlock()

	var (
		folder  = s.View().Folder()
		modURI  = s.view.modURI
		cfg     = s.config(ctx)
		options = s.view.Options()
	)
	key := modTidyKey{
		sessionID:       s.view.session.id,
		view:            folder.Filename(),
		imports:         imports,
		unsavedOverlays: overlayHash,
		gomod:           pmh.Mod().Identity().String(),
		cfg:             hashConfig(cfg),
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		ctx, done := event.Start(ctx, "cache.ModTidyHandle", tag.URI.Of(modURI))
		defer done()

		original, m, parseErrors, err := pmh.Parse(ctx)
		if err != nil || len(parseErrors) > 0 {
			return &modTidyData{
				diagnostics: parseErrors,
				err:         err,
			}
		}
		tmpURI, inv, cleanup, err := goCommandInvocation(ctx, cfg, pmh, "mod", []string{"tidy"})
		if err != nil {
			return &modTidyData{err: err}
		}
		// Keep the temporary go.mod file around long enough to parse it.
		defer cleanup()

		if _, err := packagesinternal.GetGoCmdRunner(cfg).Run(ctx, *inv); err != nil {
			return &modTidyData{err: err}
		}
		// Go directly to disk to get the temporary mod file, since it is
		// always on disk.
		tempContents, err := ioutil.ReadFile(tmpURI.Filename())
		if err != nil {
			return &modTidyData{err: err}
		}
		ideal, err := modfile.Parse(tmpURI.Filename(), tempContents, nil)
		if err != nil {
			// We do not need to worry about the temporary file's parse errors
			// since it has been "tidied".
			return &modTidyData{err: err}
		}
		// Get the dependencies that are different between the original and
		// ideal go.mod files.
		unusedDeps := make(map[string]*modfile.Require, len(original.Require))
		missingDeps := make(map[string]*modfile.Require, len(ideal.Require))
		for _, req := range original.Require {
			unusedDeps[req.Mod.Path] = req
		}
		for _, req := range ideal.Require {
			origDep := unusedDeps[req.Mod.Path]
			if origDep != nil && origDep.Indirect == req.Indirect {
				delete(unusedDeps, req.Mod.Path)
			} else {
				missingDeps[req.Mod.Path] = req
			}
		}
		diagnostics, err := modRequireErrors(pmh.Mod().URI(), original, m, missingDeps, unusedDeps, options)
		if err != nil {
			return &modTidyData{err: err}
		}
		for _, req := range missingDeps {
			if unusedDeps[req.Mod.Path] != nil {
				delete(missingDeps, req.Mod.Path)
			}
		}
		return &modTidyData{
			missingDeps: missingDeps,
			diagnostics: diagnostics,
		}
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modTidyHandle = &modTidyHandle{
		handle: h,
		pmh:    pmh,
	}
	return s.modTidyHandle, nil
}

// modRequireErrors extracts the errors that occur on the require directives.
// It checks for directness issues and unused dependencies.
func modRequireErrors(uri span.URI, parsed *modfile.File, m *protocol.ColumnMapper, missingDeps, unusedDeps map[string]*modfile.Require, options source.Options) ([]source.Error, error) {
	var errors []source.Error
	for dep, req := range unusedDeps {
		if req.Syntax == nil {
			continue
		}
		// Handle dependencies that are incorrectly labeled indirect and vice versa.
		if missingDeps[dep] != nil && req.Indirect != missingDeps[dep].Indirect {
			directErr, err := modDirectnessErrors(uri, parsed, m, req, options)
			if err != nil {
				return nil, err
			}
			errors = append(errors, directErr)
		}
		// Handle unused dependencies.
		if missingDeps[dep] == nil {
			rng, err := rangeFromPositions(uri, m, req.Syntax.Start, req.Syntax.End)
			if err != nil {
				return nil, err
			}
			edits, err := dropDependencyEdits(uri, parsed, m, req, options)
			if err != nil {
				return nil, err
			}
			errors = append(errors, source.Error{
				Category: ModTidyError,
				Message:  fmt.Sprintf("%s is not used in this module.", dep),
				Range:    rng,
				URI:      uri,
				SuggestedFixes: []source.SuggestedFix{{
					Title: fmt.Sprintf("Remove dependency: %s", dep),
					Edits: map[span.URI][]protocol.TextEdit{
						uri: edits,
					},
				}},
			})
		}
	}
	return errors, nil
}

const ModTidyError = "go mod tidy"

// modDirectnessErrors extracts errors when a dependency is labeled indirect when it should be direct and vice versa.
func modDirectnessErrors(uri span.URI, parsed *modfile.File, m *protocol.ColumnMapper, req *modfile.Require, options source.Options) (source.Error, error) {
	rng, err := rangeFromPositions(uri, m, req.Syntax.Start, req.Syntax.End)
	if err != nil {
		return source.Error{}, err
	}
	if req.Indirect {
		// If the dependency should be direct, just highlight the // indirect.
		if comments := req.Syntax.Comment(); comments != nil && len(comments.Suffix) > 0 {
			end := comments.Suffix[0].Start
			end.LineRune += len(comments.Suffix[0].Token)
			end.Byte += len([]byte(comments.Suffix[0].Token))
			rng, err = rangeFromPositions(uri, m, comments.Suffix[0].Start, end)
			if err != nil {
				return source.Error{}, err
			}
		}
		edits, err := changeDirectnessEdits(uri, parsed, m, req, false, options)
		if err != nil {
			return source.Error{}, err
		}
		return source.Error{
			Category: ModTidyError,
			Message:  fmt.Sprintf("%s should be a direct dependency.", req.Mod.Path),
			Range:    rng,
			URI:      uri,
			SuggestedFixes: []source.SuggestedFix{{
				Title: fmt.Sprintf("Make %s direct", req.Mod.Path),
				Edits: map[span.URI][]protocol.TextEdit{
					uri: edits,
				},
			}},
		}, nil
	}
	// If the dependency should be indirect, add the // indirect.
	edits, err := changeDirectnessEdits(uri, parsed, m, req, true, options)
	if err != nil {
		return source.Error{}, err
	}
	return source.Error{
		Category: ModTidyError,
		Message:  fmt.Sprintf("%s should be an indirect dependency.", req.Mod.Path),
		Range:    rng,
		URI:      uri,
		SuggestedFixes: []source.SuggestedFix{{
			Title: fmt.Sprintf("Make %s indirect", req.Mod.Path),
			Edits: map[span.URI][]protocol.TextEdit{
				uri: edits,
			},
		}},
	}, nil
}

// dropDependencyEdits gets the edits needed to remove the dependency from the go.mod file.
// As an example, this function will codify the edits needed to convert the before go.mod file to the after.
// Before:
// 	module t
//
// 	go 1.11
//
// 	require golang.org/x/mod v0.1.1-0.20191105210325-c90efee705ee
// After:
// 	module t
//
// 	go 1.11
func dropDependencyEdits(uri span.URI, parsed *modfile.File, m *protocol.ColumnMapper, req *modfile.Require, options source.Options) ([]protocol.TextEdit, error) {
	if err := parsed.DropRequire(req.Mod.Path); err != nil {
		return nil, err
	}
	parsed.Cleanup()
	newContents, err := parsed.Format()
	if err != nil {
		return nil, err
	}
	// Reset the *modfile.File back to before we dropped the dependency.
	parsed.AddNewRequire(req.Mod.Path, req.Mod.Version, req.Indirect)
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(uri, string(m.Content), string(newContents))
	edits, err := source.ToProtocolEdits(m, diff)
	if err != nil {
		return nil, err
	}
	return edits, nil
}

// changeDirectnessEdits gets the edits needed to change an indirect dependency to direct and vice versa.
// As an example, this function will codify the edits needed to convert the before go.mod file to the after.
// Before:
// 	module t
//
// 	go 1.11
//
// 	require golang.org/x/mod v0.1.1-0.20191105210325-c90efee705ee
// After:
// 	module t
//
// 	go 1.11
//
// 	require golang.org/x/mod v0.1.1-0.20191105210325-c90efee705ee // indirect
func changeDirectnessEdits(uri span.URI, parsed *modfile.File, m *protocol.ColumnMapper, req *modfile.Require, indirect bool, options source.Options) ([]protocol.TextEdit, error) {
	var newReq []*modfile.Require
	prevIndirect := false
	// Change the directness in the matching require statement.
	for _, r := range parsed.Require {
		if req.Mod.Path == r.Mod.Path {
			prevIndirect = req.Indirect
			req.Indirect = indirect
		}
		newReq = append(newReq, r)
	}
	parsed.SetRequire(newReq)
	parsed.Cleanup()
	newContents, err := parsed.Format()
	if err != nil {
		return nil, err
	}
	// Change the dependency back to the way it was before we got the newContents.
	for _, r := range parsed.Require {
		if req.Mod.Path == r.Mod.Path {
			req.Indirect = prevIndirect
		}
		newReq = append(newReq, r)
	}
	parsed.SetRequire(newReq)
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(uri, string(m.Content), string(newContents))
	edits, err := source.ToProtocolEdits(m, diff)
	if err != nil {
		return nil, err
	}
	return edits, nil
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
