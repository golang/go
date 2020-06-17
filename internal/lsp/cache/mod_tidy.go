// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"io/ioutil"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type modTidyHandle struct {
	handle *memoize.Handle

	file source.FileHandle
	cfg  *packages.Config
}

type modTidyData struct {
	memoize.NoCopy

	// fh is the file handle for the original go.mod file.
	fh source.FileHandle

	// parsed contains the parsed contents that are used to diff with
	// the ideal contents.
	parsed *modfile.File

	// m is the column mapper for the original go.mod file.
	m *protocol.ColumnMapper

	// parseErrors are the errors that arise when we diff between a user's go.mod
	// and the "tidied" go.mod.
	parseErrors []source.Error

	// ideal contains the parsed contents for the go.mod file
	// after it has been "tidied".
	ideal *modfile.File

	// unusedDeps is the map containing the dependencies that are left after
	// removing the ones that are identical in the original and ideal go.mods.
	unusedDeps map[string]*modfile.Require

	// missingDeps is the map containing that are missing from the original
	// go.mod, but present in the ideal go.mod.
	missingDeps map[string]*modfile.Require

	// err is any error that occurs while we are calculating the parseErrors.
	err error
}

func (mh *modTidyHandle) String() string {
	return mh.File().URI().Filename()
}

func (mh *modTidyHandle) File() source.FileHandle {
	return mh.file
}

func (mh *modTidyHandle) Tidy(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, map[string]*modfile.Require, []source.Error, error) {
	v := mh.handle.Get(ctx)
	if v == nil {
		return nil, nil, nil, nil, errors.Errorf("no tidied file for %s", mh.File().URI())
	}
	data := v.(*modTidyData)
	return data.parsed, data.m, data.missingDeps, data.parseErrors, data.err
}

func (s *snapshot) ModTidyHandle(ctx context.Context, modFH source.FileHandle) (source.ModTidyHandle, error) {
	if handle := s.getModTidyHandle(); handle != nil {
		return handle, nil
	}
	var sumFH source.FileHandle
	if s.view.sumURI != "" {
		var err error
		sumFH, err = s.GetFile(ctx, s.view.sumURI)
		if err != nil {
			return nil, err
		}
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
		options = s.View().Options()
		folder  = s.View().Folder()
		modURI  = s.view.modURI
		tmpMod  = s.view.tmpMod
	)

	cfg := s.config(ctx)
	key := modTidyKey{
		sessionID:       s.view.session.id,
		view:            folder.Filename(),
		imports:         imports,
		unsavedOverlays: overlayHash,
		gomod:           modFH.Identity().String(),
		cfg:             hashConfig(cfg),
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		ctx, done := event.Start(ctx, "cache.ModTidyHandle", tag.URI.Of(modURI))
		defer done()

		// Do nothing if the -modfile flag is disabled or if the given go.mod
		// is outside of our view.
		if modURI != modFH.URI() || !tmpMod {
			return &modTidyData{}
		}

		contents, err := modFH.Read()
		if err != nil {
			return &modTidyData{err: err}
		}
		realMapper := &protocol.ColumnMapper{
			URI:       modURI,
			Converter: span.NewContentConverter(modURI.Filename(), contents),
			Content:   contents,
		}
		origParsedFile, err := modfile.Parse(modURI.Filename(), contents, nil)
		if err != nil {
			if parseErr, err := extractModParseErrors(ctx, modURI, realMapper, err, contents); err == nil {
				return &modTidyData{parseErrors: []source.Error{parseErr}}
			}
			return &modTidyData{err: err}
		}
		tmpURI, inv, cleanup, err := goCommandInvocation(ctx, cfg, modFH, sumFH, "mod", []string{"tidy"})
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
		idealParsedFile, err := modfile.Parse(tmpURI.Filename(), tempContents, nil)
		if err != nil {
			// We do not need to worry about the temporary file's parse errors
			// since it has been "tidied".
			return &modTidyData{err: err}
		}

		data := &modTidyData{
			fh:          modFH,
			parsed:      origParsedFile,
			m:           realMapper,
			ideal:       idealParsedFile,
			unusedDeps:  make(map[string]*modfile.Require, len(origParsedFile.Require)),
			missingDeps: make(map[string]*modfile.Require, len(idealParsedFile.Require)),
		}
		// Get the dependencies that are different between the original and
		// ideal go.mod files.
		for _, req := range origParsedFile.Require {
			data.unusedDeps[req.Mod.Path] = req
		}
		for _, req := range idealParsedFile.Require {
			origDep := data.unusedDeps[req.Mod.Path]
			if origDep != nil && origDep.Indirect == req.Indirect {
				delete(data.unusedDeps, req.Mod.Path)
			} else {
				data.missingDeps[req.Mod.Path] = req
			}
		}
		data.parseErrors, data.err = modRequireErrors(options, data)

		for _, req := range data.missingDeps {
			if data.unusedDeps[req.Mod.Path] != nil {
				delete(data.missingDeps, req.Mod.Path)
			}
		}
		return data
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modTidyHandle = &modTidyHandle{
		handle: h,
		file:   modFH,
		cfg:    cfg,
	}
	return s.modTidyHandle, nil
}

// extractModParseErrors processes the raw errors returned by modfile.Parse,
// extracting the filenames and line numbers that correspond to the errors.
func extractModParseErrors(ctx context.Context, uri span.URI, m *protocol.ColumnMapper, parseErr error, content []byte) (source.Error, error) {
	re := regexp.MustCompile(`.*:([\d]+): (.+)`)
	matches := re.FindStringSubmatch(strings.TrimSpace(parseErr.Error()))
	if len(matches) < 3 {
		event.Error(ctx, "could not parse golang/x/mod error message", parseErr)
		return source.Error{}, parseErr
	}
	line, err := strconv.Atoi(matches[1])
	if err != nil {
		return source.Error{}, parseErr
	}
	lines := strings.Split(string(content), "\n")
	if len(lines) <= line {
		return source.Error{}, errors.Errorf("could not parse goland/x/mod error message, line number out of range")
	}
	// The error returned from the modfile package only returns a line number,
	// so we assume that the diagnostic should be for the entire line.
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

// modRequireErrors extracts the errors that occur on the require directives.
// It checks for directness issues and unused dependencies.
func modRequireErrors(options source.Options, data *modTidyData) ([]source.Error, error) {
	var errors []source.Error
	for dep, req := range data.unusedDeps {
		if req.Syntax == nil {
			continue
		}
		// Handle dependencies that are incorrectly labeled indirect and vice versa.
		if data.missingDeps[dep] != nil && req.Indirect != data.missingDeps[dep].Indirect {
			directErr, err := modDirectnessErrors(options, data, req)
			if err != nil {
				return nil, err
			}
			errors = append(errors, directErr)
		}
		// Handle unused dependencies.
		if data.missingDeps[dep] == nil {
			rng, err := rangeFromPositions(data.fh.URI(), data.m, req.Syntax.Start, req.Syntax.End)
			if err != nil {
				return nil, err
			}
			edits, err := dropDependencyEdits(options, data, req)
			if err != nil {
				return nil, err
			}
			errors = append(errors, source.Error{
				Category: ModTidyError,
				Message:  fmt.Sprintf("%s is not used in this module.", dep),
				Range:    rng,
				URI:      data.fh.URI(),
				SuggestedFixes: []source.SuggestedFix{{
					Title: fmt.Sprintf("Remove dependency: %s", dep),
					Edits: map[span.URI][]protocol.TextEdit{data.fh.URI(): edits},
				}},
			})
		}
	}
	return errors, nil
}

// modDirectnessErrors extracts errors when a dependency is labeled indirect when it should be direct and vice versa.
func modDirectnessErrors(options source.Options, data *modTidyData, req *modfile.Require) (source.Error, error) {
	rng, err := rangeFromPositions(data.fh.URI(), data.m, req.Syntax.Start, req.Syntax.End)
	if err != nil {
		return source.Error{}, err
	}
	if req.Indirect {
		// If the dependency should be direct, just highlight the // indirect.
		if comments := req.Syntax.Comment(); comments != nil && len(comments.Suffix) > 0 {
			end := comments.Suffix[0].Start
			end.LineRune += len(comments.Suffix[0].Token)
			end.Byte += len([]byte(comments.Suffix[0].Token))
			rng, err = rangeFromPositions(data.fh.URI(), data.m, comments.Suffix[0].Start, end)
			if err != nil {
				return source.Error{}, err
			}
		}
		edits, err := changeDirectnessEdits(options, data, req, false)
		if err != nil {
			return source.Error{}, err
		}
		return source.Error{
			Category: ModTidyError,
			Message:  fmt.Sprintf("%s should be a direct dependency.", req.Mod.Path),
			Range:    rng,
			URI:      data.fh.URI(),
			SuggestedFixes: []source.SuggestedFix{{
				Title: fmt.Sprintf("Make %s direct", req.Mod.Path),
				Edits: map[span.URI][]protocol.TextEdit{data.fh.URI(): edits},
			}},
		}, nil
	}
	// If the dependency should be indirect, add the // indirect.
	edits, err := changeDirectnessEdits(options, data, req, true)
	if err != nil {
		return source.Error{}, err
	}
	return source.Error{
		Category: ModTidyError,
		Message:  fmt.Sprintf("%s should be an indirect dependency.", req.Mod.Path),
		Range:    rng,
		URI:      data.fh.URI(),
		SuggestedFixes: []source.SuggestedFix{{
			Title: fmt.Sprintf("Make %s indirect", req.Mod.Path),
			Edits: map[span.URI][]protocol.TextEdit{data.fh.URI(): edits},
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
func dropDependencyEdits(options source.Options, data *modTidyData, req *modfile.Require) ([]protocol.TextEdit, error) {
	if err := data.parsed.DropRequire(req.Mod.Path); err != nil {
		return nil, err
	}
	data.parsed.Cleanup()
	newContents, err := data.parsed.Format()
	if err != nil {
		return nil, err
	}
	// Reset the *modfile.File back to before we dropped the dependency.
	data.parsed.AddNewRequire(req.Mod.Path, req.Mod.Version, req.Indirect)
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(data.fh.URI(), string(data.m.Content), string(newContents))
	edits, err := source.ToProtocolEdits(data.m, diff)
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
func changeDirectnessEdits(options source.Options, data *modTidyData, req *modfile.Require, indirect bool) ([]protocol.TextEdit, error) {
	var newReq []*modfile.Require
	prevIndirect := false
	// Change the directness in the matching require statement.
	for _, r := range data.parsed.Require {
		if req.Mod.Path == r.Mod.Path {
			prevIndirect = req.Indirect
			req.Indirect = indirect
		}
		newReq = append(newReq, r)
	}
	data.parsed.SetRequire(newReq)
	data.parsed.Cleanup()
	newContents, err := data.parsed.Format()
	if err != nil {
		return nil, err
	}
	// Change the dependency back to the way it was before we got the newContents.
	for _, r := range data.parsed.Require {
		if req.Mod.Path == r.Mod.Path {
			req.Indirect = prevIndirect
		}
		newReq = append(newReq, r)
	}
	data.parsed.SetRequire(newReq)
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(data.fh.URI(), string(data.m.Content), string(newContents))
	edits, err := source.ToProtocolEdits(data.m, diff)
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
