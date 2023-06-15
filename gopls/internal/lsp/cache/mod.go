// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/memoize"
)

// ParseMod parses a go.mod file, using a cache. It may return partial results and an error.
func (s *snapshot) ParseMod(ctx context.Context, fh source.FileHandle) (*source.ParsedModule, error) {
	uri := fh.URI()

	s.mu.Lock()
	entry, hit := s.parseModHandles.Get(uri)
	s.mu.Unlock()

	type parseModKey source.FileIdentity
	type parseModResult struct {
		parsed *source.ParsedModule
		err    error
	}

	// cache miss?
	if !hit {
		promise, release := s.store.Promise(parseModKey(fh.FileIdentity()), func(ctx context.Context, _ interface{}) interface{} {
			parsed, err := parseModImpl(ctx, fh)
			return parseModResult{parsed, err}
		})

		entry = promise
		s.mu.Lock()
		s.parseModHandles.Set(uri, entry, func(_, _ interface{}) { release() })
		s.mu.Unlock()
	}

	// Await result.
	v, err := s.awaitPromise(ctx, entry.(*memoize.Promise))
	if err != nil {
		return nil, err
	}
	res := v.(parseModResult)
	return res.parsed, res.err
}

// parseModImpl parses the go.mod file whose name and contents are in fh.
// It may return partial results and an error.
func parseModImpl(ctx context.Context, fh source.FileHandle) (*source.ParsedModule, error) {
	_, done := event.Start(ctx, "cache.ParseMod", tag.URI.Of(fh.URI()))
	defer done()

	contents, err := fh.Content()
	if err != nil {
		return nil, err
	}
	m := protocol.NewMapper(fh.URI(), contents)
	file, parseErr := modfile.Parse(fh.URI().Filename(), contents, nil)
	// Attempt to convert the error to a standardized parse error.
	var parseErrors []*source.Diagnostic
	if parseErr != nil {
		mfErrList, ok := parseErr.(modfile.ErrorList)
		if !ok {
			return nil, fmt.Errorf("unexpected parse error type %v", parseErr)
		}
		for _, mfErr := range mfErrList {
			rng, err := m.OffsetRange(mfErr.Pos.Byte, mfErr.Pos.Byte)
			if err != nil {
				return nil, err
			}
			parseErrors = append(parseErrors, &source.Diagnostic{
				URI:      fh.URI(),
				Range:    rng,
				Severity: protocol.SeverityError,
				Source:   source.ParseError,
				Message:  mfErr.Err.Error(),
			})
		}
	}
	return &source.ParsedModule{
		URI:         fh.URI(),
		Mapper:      m,
		File:        file,
		ParseErrors: parseErrors,
	}, parseErr
}

// ParseWork parses a go.work file, using a cache. It may return partial results and an error.
// TODO(adonovan): move to new work.go file.
func (s *snapshot) ParseWork(ctx context.Context, fh source.FileHandle) (*source.ParsedWorkFile, error) {
	uri := fh.URI()

	s.mu.Lock()
	entry, hit := s.parseWorkHandles.Get(uri)
	s.mu.Unlock()

	type parseWorkKey source.FileIdentity
	type parseWorkResult struct {
		parsed *source.ParsedWorkFile
		err    error
	}

	// cache miss?
	if !hit {
		handle, release := s.store.Promise(parseWorkKey(fh.FileIdentity()), func(ctx context.Context, _ interface{}) interface{} {
			parsed, err := parseWorkImpl(ctx, fh)
			return parseWorkResult{parsed, err}
		})

		entry = handle
		s.mu.Lock()
		s.parseWorkHandles.Set(uri, entry, func(_, _ interface{}) { release() })
		s.mu.Unlock()
	}

	// Await result.
	v, err := s.awaitPromise(ctx, entry.(*memoize.Promise))
	if err != nil {
		return nil, err
	}
	res := v.(parseWorkResult)
	return res.parsed, res.err
}

// parseWorkImpl parses a go.work file. It may return partial results and an error.
func parseWorkImpl(ctx context.Context, fh source.FileHandle) (*source.ParsedWorkFile, error) {
	_, done := event.Start(ctx, "cache.ParseWork", tag.URI.Of(fh.URI()))
	defer done()

	content, err := fh.Content()
	if err != nil {
		return nil, err
	}
	m := protocol.NewMapper(fh.URI(), content)
	file, parseErr := modfile.ParseWork(fh.URI().Filename(), content, nil)
	// Attempt to convert the error to a standardized parse error.
	var parseErrors []*source.Diagnostic
	if parseErr != nil {
		mfErrList, ok := parseErr.(modfile.ErrorList)
		if !ok {
			return nil, fmt.Errorf("unexpected parse error type %v", parseErr)
		}
		for _, mfErr := range mfErrList {
			rng, err := m.OffsetRange(mfErr.Pos.Byte, mfErr.Pos.Byte)
			if err != nil {
				return nil, err
			}
			parseErrors = append(parseErrors, &source.Diagnostic{
				URI:      fh.URI(),
				Range:    rng,
				Severity: protocol.SeverityError,
				Source:   source.ParseError,
				Message:  mfErr.Err.Error(),
			})
		}
	}
	return &source.ParsedWorkFile{
		URI:         fh.URI(),
		Mapper:      m,
		File:        file,
		ParseErrors: parseErrors,
	}, parseErr
}

// goSum reads the go.sum file for the go.mod file at modURI, if it exists. If
// it doesn't exist, it returns nil.
func (s *snapshot) goSum(ctx context.Context, modURI span.URI) []byte {
	// Get the go.sum file, either from the snapshot or directly from the
	// cache. Avoid (*snapshot).ReadFile here, as we don't want to add
	// nonexistent file handles to the snapshot if the file does not exist.
	//
	// TODO(rfindley): but that's not right. Changes to sum files should
	// invalidate content, even if it's nonexistent content.
	sumURI := span.URIFromPath(sumFilename(modURI))
	var sumFH source.FileHandle = s.FindFile(sumURI)
	if sumFH == nil {
		var err error
		sumFH, err = s.view.fs.ReadFile(ctx, sumURI)
		if err != nil {
			return nil
		}
	}
	content, err := sumFH.Content()
	if err != nil {
		return nil
	}
	return content
}

func sumFilename(modURI span.URI) string {
	return strings.TrimSuffix(modURI.Filename(), ".mod") + ".sum"
}

// ModWhy returns the "go mod why" result for each module named in a
// require statement in the go.mod file.
// TODO(adonovan): move to new mod_why.go file.
func (s *snapshot) ModWhy(ctx context.Context, fh source.FileHandle) (map[string]string, error) {
	uri := fh.URI()

	if s.View().FileKind(fh) != source.Mod {
		return nil, fmt.Errorf("%s is not a go.mod file", uri)
	}

	s.mu.Lock()
	entry, hit := s.modWhyHandles.Get(uri)
	s.mu.Unlock()

	type modWhyResult struct {
		why map[string]string
		err error
	}

	// cache miss?
	if !hit {
		handle := memoize.NewPromise("modWhy", func(ctx context.Context, arg interface{}) interface{} {
			why, err := modWhyImpl(ctx, arg.(*snapshot), fh)
			return modWhyResult{why, err}
		})

		entry = handle
		s.mu.Lock()
		s.modWhyHandles.Set(uri, entry, nil)
		s.mu.Unlock()
	}

	// Await result.
	v, err := s.awaitPromise(ctx, entry.(*memoize.Promise))
	if err != nil {
		return nil, err
	}
	res := v.(modWhyResult)
	return res.why, res.err
}

// modWhyImpl returns the result of "go mod why -m" on the specified go.mod file.
func modWhyImpl(ctx context.Context, snapshot *snapshot, fh source.FileHandle) (map[string]string, error) {
	ctx, done := event.Start(ctx, "cache.ModWhy", tag.URI.Of(fh.URI()))
	defer done()

	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		return nil, err
	}
	// No requires to explain.
	if len(pm.File.Require) == 0 {
		return nil, nil // empty result
	}
	// Run `go mod why` on all the dependencies.
	inv := &gocommand.Invocation{
		Verb:       "mod",
		Args:       []string{"why", "-m"},
		WorkingDir: filepath.Dir(fh.URI().Filename()),
	}
	for _, req := range pm.File.Require {
		inv.Args = append(inv.Args, req.Mod.Path)
	}
	stdout, err := snapshot.RunGoCommandDirect(ctx, source.Normal, inv)
	if err != nil {
		return nil, err
	}
	whyList := strings.Split(stdout.String(), "\n\n")
	if len(whyList) != len(pm.File.Require) {
		return nil, fmt.Errorf("mismatched number of results: got %v, want %v", len(whyList), len(pm.File.Require))
	}
	why := make(map[string]string, len(pm.File.Require))
	for i, req := range pm.File.Require {
		why[req.Mod.Path] = whyList[i]
	}
	return why, nil
}

// extractGoCommandErrors tries to parse errors that come from the go command
// and shape them into go.mod diagnostics.
// TODO: rename this to 'load errors'
func (s *snapshot) extractGoCommandErrors(ctx context.Context, goCmdError error) []*source.Diagnostic {
	if goCmdError == nil {
		return nil
	}

	type locatedErr struct {
		loc protocol.Location
		msg string
	}
	diagLocations := map[*source.ParsedModule]locatedErr{}
	backupDiagLocations := map[*source.ParsedModule]locatedErr{}

	// If moduleErrs is non-nil, go command errors are scoped to specific
	// modules.
	var moduleErrs *moduleErrorMap
	_ = errors.As(goCmdError, &moduleErrs)

	// Match the error against all the mod files in the workspace.
	for _, uri := range s.ModFiles() {
		fh, err := s.ReadFile(ctx, uri)
		if err != nil {
			event.Error(ctx, "getting modfile for Go command error", err)
			continue
		}
		pm, err := s.ParseMod(ctx, fh)
		if err != nil {
			// Parsing errors are reported elsewhere
			return nil
		}
		var msgs []string // error messages to consider
		if moduleErrs != nil {
			if pm.File.Module != nil {
				for _, mes := range moduleErrs.errs[pm.File.Module.Mod.Path] {
					msgs = append(msgs, mes.Error())
				}
			}
		} else {
			msgs = append(msgs, goCmdError.Error())
		}
		for _, msg := range msgs {
			if strings.Contains(goCmdError.Error(), "errors parsing go.mod") {
				// The go command emits parse errors for completely invalid go.mod files.
				// Those are reported by our own diagnostics and can be ignored here.
				// As of writing, we are not aware of any other errors that include
				// file/position information, so don't even try to find it.
				continue
			}
			loc, found, err := s.matchErrorToModule(ctx, pm, msg)
			if err != nil {
				event.Error(ctx, "matching error to module", err)
				continue
			}
			le := locatedErr{
				loc: loc,
				msg: msg,
			}
			if found {
				diagLocations[pm] = le
			} else {
				backupDiagLocations[pm] = le
			}
		}
	}

	// If we didn't find any good matches, assign diagnostics to all go.mod files.
	if len(diagLocations) == 0 {
		diagLocations = backupDiagLocations
	}

	var srcErrs []*source.Diagnostic
	for pm, le := range diagLocations {
		diag, err := s.goCommandDiagnostic(pm, le.loc, le.msg)
		if err != nil {
			event.Error(ctx, "building go command diagnostic", err)
			continue
		}
		srcErrs = append(srcErrs, diag)
	}
	return srcErrs
}

var moduleVersionInErrorRe = regexp.MustCompile(`[:\s]([+-._~0-9A-Za-z]+)@([+-._~0-9A-Za-z]+)[:\s]`)

// matchErrorToModule matches a go command error message to a go.mod file.
// Some examples:
//
//	example.com@v1.2.2: reading example.com/@v/v1.2.2.mod: no such file or directory
//	go: github.com/cockroachdb/apd/v2@v2.0.72: reading github.com/cockroachdb/apd/go.mod at revision v2.0.72: unknown revision v2.0.72
//	go: example.com@v1.2.3 requires\n\trandom.org@v1.2.3: parsing go.mod:\n\tmodule declares its path as: bob.org\n\tbut was required as: random.org
//
// It returns the location of a reference to the one of the modules and true
// if one exists. If none is found it returns a fallback location and false.
func (s *snapshot) matchErrorToModule(ctx context.Context, pm *source.ParsedModule, goCmdError string) (protocol.Location, bool, error) {
	var reference *modfile.Line
	matches := moduleVersionInErrorRe.FindAllStringSubmatch(goCmdError, -1)

	for i := len(matches) - 1; i >= 0; i-- {
		ver := module.Version{Path: matches[i][1], Version: matches[i][2]}
		if err := module.Check(ver.Path, ver.Version); err != nil {
			continue
		}
		reference = findModuleReference(pm.File, ver)
		if reference != nil {
			break
		}
	}

	if reference == nil {
		// No match for the module path was found in the go.mod file.
		// Show the error on the module declaration, if one exists, or
		// just the first line of the file.
		var start, end int
		if pm.File.Module != nil && pm.File.Module.Syntax != nil {
			syntax := pm.File.Module.Syntax
			start, end = syntax.Start.Byte, syntax.End.Byte
		}
		loc, err := pm.Mapper.OffsetLocation(start, end)
		return loc, false, err
	}

	loc, err := pm.Mapper.OffsetLocation(reference.Start.Byte, reference.End.Byte)
	return loc, true, err
}

// goCommandDiagnostic creates a diagnostic for a given go command error.
func (s *snapshot) goCommandDiagnostic(pm *source.ParsedModule, loc protocol.Location, goCmdError string) (*source.Diagnostic, error) {
	matches := moduleVersionInErrorRe.FindAllStringSubmatch(goCmdError, -1)
	var innermost *module.Version
	for i := len(matches) - 1; i >= 0; i-- {
		ver := module.Version{Path: matches[i][1], Version: matches[i][2]}
		if err := module.Check(ver.Path, ver.Version); err != nil {
			continue
		}
		innermost = &ver
		break
	}

	switch {
	case strings.Contains(goCmdError, "inconsistent vendoring"):
		cmd, err := command.NewVendorCommand("Run go mod vendor", command.URIArg{URI: protocol.URIFromSpanURI(pm.URI)})
		if err != nil {
			return nil, err
		}
		return &source.Diagnostic{
			URI:      pm.URI,
			Range:    loc.Range,
			Severity: protocol.SeverityError,
			Source:   source.ListError,
			Message: `Inconsistent vendoring detected. Please re-run "go mod vendor".
See https://github.com/golang/go/issues/39164 for more detail on this issue.`,
			SuggestedFixes: []source.SuggestedFix{source.SuggestedFixFromCommand(cmd, protocol.QuickFix)},
		}, nil

	case strings.Contains(goCmdError, "updates to go.sum needed"), strings.Contains(goCmdError, "missing go.sum entry"):
		var args []protocol.DocumentURI
		for _, uri := range s.ModFiles() {
			args = append(args, protocol.URIFromSpanURI(uri))
		}
		tidyCmd, err := command.NewTidyCommand("Run go mod tidy", command.URIArgs{URIs: args})
		if err != nil {
			return nil, err
		}
		updateCmd, err := command.NewUpdateGoSumCommand("Update go.sum", command.URIArgs{URIs: args})
		if err != nil {
			return nil, err
		}
		msg := "go.sum is out of sync with go.mod. Please update it by applying the quick fix."
		if innermost != nil {
			msg = fmt.Sprintf("go.sum is out of sync with go.mod: entry for %v is missing. Please updating it by applying the quick fix.", innermost)
		}
		return &source.Diagnostic{
			URI:      pm.URI,
			Range:    loc.Range,
			Severity: protocol.SeverityError,
			Source:   source.ListError,
			Message:  msg,
			SuggestedFixes: []source.SuggestedFix{
				source.SuggestedFixFromCommand(tidyCmd, protocol.QuickFix),
				source.SuggestedFixFromCommand(updateCmd, protocol.QuickFix),
			},
		}, nil
	case strings.Contains(goCmdError, "disabled by GOPROXY=off") && innermost != nil:
		title := fmt.Sprintf("Download %v@%v", innermost.Path, innermost.Version)
		cmd, err := command.NewAddDependencyCommand(title, command.DependencyArgs{
			URI:        protocol.URIFromSpanURI(pm.URI),
			AddRequire: false,
			GoCmdArgs:  []string{fmt.Sprintf("%v@%v", innermost.Path, innermost.Version)},
		})
		if err != nil {
			return nil, err
		}
		return &source.Diagnostic{
			URI:            pm.URI,
			Range:          loc.Range,
			Severity:       protocol.SeverityError,
			Message:        fmt.Sprintf("%v@%v has not been downloaded", innermost.Path, innermost.Version),
			Source:         source.ListError,
			SuggestedFixes: []source.SuggestedFix{source.SuggestedFixFromCommand(cmd, protocol.QuickFix)},
		}, nil
	default:
		return &source.Diagnostic{
			URI:      pm.URI,
			Range:    loc.Range,
			Severity: protocol.SeverityError,
			Source:   source.ListError,
			Message:  goCmdError,
		}, nil
	}
}

func findModuleReference(mf *modfile.File, ver module.Version) *modfile.Line {
	for _, req := range mf.Require {
		if req.Mod == ver {
			return req.Syntax
		}
	}
	for _, ex := range mf.Exclude {
		if ex.Mod == ver {
			return ex.Syntax
		}
	}
	for _, rep := range mf.Replace {
		if rep.New == ver || rep.Old == ver {
			return rep.Syntax
		}
	}
	return nil
}
