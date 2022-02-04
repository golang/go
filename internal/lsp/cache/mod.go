// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
)

type parseModHandle struct {
	handle *memoize.Handle
}

type parseModData struct {
	parsed *source.ParsedModule

	// err is any error encountered while parsing the file.
	err error
}

func (mh *parseModHandle) parse(ctx context.Context, snapshot *snapshot) (*source.ParsedModule, error) {
	v, err := mh.handle.Get(ctx, snapshot.generation, snapshot)
	if err != nil {
		return nil, err
	}
	data := v.(*parseModData)
	return data.parsed, data.err
}

func (s *snapshot) ParseMod(ctx context.Context, modFH source.FileHandle) (*source.ParsedModule, error) {
	if handle := s.getParseModHandle(modFH.URI()); handle != nil {
		return handle.parse(ctx, s)
	}
	h := s.generation.Bind(modFH.FileIdentity(), func(ctx context.Context, _ memoize.Arg) interface{} {
		_, done := event.Start(ctx, "cache.ParseModHandle", tag.URI.Of(modFH.URI()))
		defer done()

		contents, err := modFH.Read()
		if err != nil {
			return &parseModData{err: err}
		}
		m := &protocol.ColumnMapper{
			URI:       modFH.URI(),
			Converter: span.NewContentConverter(modFH.URI().Filename(), contents),
			Content:   contents,
		}
		file, parseErr := modfile.Parse(modFH.URI().Filename(), contents, nil)
		// Attempt to convert the error to a standardized parse error.
		var parseErrors []*source.Diagnostic
		if parseErr != nil {
			mfErrList, ok := parseErr.(modfile.ErrorList)
			if !ok {
				return &parseModData{err: fmt.Errorf("unexpected parse error type %v", parseErr)}
			}
			for _, mfErr := range mfErrList {
				rng, err := rangeFromPositions(m, mfErr.Pos, mfErr.Pos)
				if err != nil {
					return &parseModData{err: err}
				}
				parseErrors = []*source.Diagnostic{{
					URI:      modFH.URI(),
					Range:    rng,
					Severity: protocol.SeverityError,
					Source:   source.ParseError,
					Message:  mfErr.Err.Error(),
				}}
			}
		}
		return &parseModData{
			parsed: &source.ParsedModule{
				URI:         modFH.URI(),
				Mapper:      m,
				File:        file,
				ParseErrors: parseErrors,
			},
			err: parseErr,
		}
	}, nil)

	pmh := &parseModHandle{handle: h}
	s.mu.Lock()
	s.parseModHandles[modFH.URI()] = pmh
	s.mu.Unlock()

	return pmh.parse(ctx, s)
}

type parseWorkHandle struct {
	handle *memoize.Handle
}

type parseWorkData struct {
	parsed *source.ParsedWorkFile

	// err is any error encountered while parsing the file.
	err error
}

func (mh *parseWorkHandle) parse(ctx context.Context, snapshot *snapshot) (*source.ParsedWorkFile, error) {
	v, err := mh.handle.Get(ctx, snapshot.generation, snapshot)
	if err != nil {
		return nil, err
	}
	data := v.(*parseWorkData)
	return data.parsed, data.err
}

func (s *snapshot) ParseWork(ctx context.Context, modFH source.FileHandle) (*source.ParsedWorkFile, error) {
	if handle := s.getParseWorkHandle(modFH.URI()); handle != nil {
		return handle.parse(ctx, s)
	}
	h := s.generation.Bind(modFH.FileIdentity(), func(ctx context.Context, _ memoize.Arg) interface{} {
		_, done := event.Start(ctx, "cache.ParseModHandle", tag.URI.Of(modFH.URI()))
		defer done()

		contents, err := modFH.Read()
		if err != nil {
			return &parseModData{err: err}
		}
		m := &protocol.ColumnMapper{
			URI:       modFH.URI(),
			Converter: span.NewContentConverter(modFH.URI().Filename(), contents),
			Content:   contents,
		}
		file, parseErr := modfile.ParseWork(modFH.URI().Filename(), contents, nil)
		// Attempt to convert the error to a standardized parse error.
		var parseErrors []*source.Diagnostic
		if parseErr != nil {
			mfErrList, ok := parseErr.(modfile.ErrorList)
			if !ok {
				return &parseModData{err: fmt.Errorf("unexpected parse error type %v", parseErr)}
			}
			for _, mfErr := range mfErrList {
				rng, err := rangeFromPositions(m, mfErr.Pos, mfErr.Pos)
				if err != nil {
					return &parseModData{err: err}
				}
				parseErrors = []*source.Diagnostic{{
					URI:      modFH.URI(),
					Range:    rng,
					Severity: protocol.SeverityError,
					Source:   source.ParseError,
					Message:  mfErr.Err.Error(),
				}}
			}
		}
		return &parseWorkData{
			parsed: &source.ParsedWorkFile{
				URI:         modFH.URI(),
				Mapper:      m,
				File:        file,
				ParseErrors: parseErrors,
			},
			err: parseErr,
		}
	}, nil)

	pwh := &parseWorkHandle{handle: h}
	s.mu.Lock()
	s.parseWorkHandles[modFH.URI()] = pwh
	s.mu.Unlock()

	return pwh.parse(ctx, s)
}

// goSum reads the go.sum file for the go.mod file at modURI, if it exists. If
// it doesn't exist, it returns nil.
func (s *snapshot) goSum(ctx context.Context, modURI span.URI) []byte {
	// Get the go.sum file, either from the snapshot or directly from the
	// cache. Avoid (*snapshot).GetFile here, as we don't want to add
	// nonexistent file handles to the snapshot if the file does not exist.
	sumURI := span.URIFromPath(sumFilename(modURI))
	var sumFH source.FileHandle = s.FindFile(sumURI)
	if sumFH == nil {
		var err error
		sumFH, err = s.view.session.cache.getFile(ctx, sumURI)
		if err != nil {
			return nil
		}
	}
	content, err := sumFH.Read()
	if err != nil {
		return nil
	}
	return content
}

func sumFilename(modURI span.URI) string {
	return strings.TrimSuffix(modURI.Filename(), ".mod") + ".sum"
}

// modKey is uniquely identifies cached data for `go mod why` or dependencies
// to upgrade.
type modKey struct {
	sessionID, env, view string
	mod                  source.FileIdentity
	verb                 modAction
}

type modAction int

const (
	why modAction = iota
	upgrade
)

type modWhyHandle struct {
	handle *memoize.Handle
}

type modWhyData struct {
	// why keeps track of the `go mod why` results for each require statement
	// in the go.mod file.
	why map[string]string

	err error
}

func (mwh *modWhyHandle) why(ctx context.Context, snapshot *snapshot) (map[string]string, error) {
	v, err := mwh.handle.Get(ctx, snapshot.generation, snapshot)
	if err != nil {
		return nil, err
	}
	data := v.(*modWhyData)
	return data.why, data.err
}

func (s *snapshot) ModWhy(ctx context.Context, fh source.FileHandle) (map[string]string, error) {
	if s.View().FileKind(fh) != source.Mod {
		return nil, fmt.Errorf("%s is not a go.mod file", fh.URI())
	}
	if handle := s.getModWhyHandle(fh.URI()); handle != nil {
		return handle.why(ctx, s)
	}
	key := modKey{
		sessionID: s.view.session.id,
		env:       hashEnv(s),
		mod:       fh.FileIdentity(),
		view:      s.view.rootURI.Filename(),
		verb:      why,
	}
	h := s.generation.Bind(key, func(ctx context.Context, arg memoize.Arg) interface{} {
		ctx, done := event.Start(ctx, "cache.ModWhyHandle", tag.URI.Of(fh.URI()))
		defer done()

		snapshot := arg.(*snapshot)

		pm, err := snapshot.ParseMod(ctx, fh)
		if err != nil {
			return &modWhyData{err: err}
		}
		// No requires to explain.
		if len(pm.File.Require) == 0 {
			return &modWhyData{}
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
			return &modWhyData{err: err}
		}
		whyList := strings.Split(stdout.String(), "\n\n")
		if len(whyList) != len(pm.File.Require) {
			return &modWhyData{
				err: fmt.Errorf("mismatched number of results: got %v, want %v", len(whyList), len(pm.File.Require)),
			}
		}
		why := make(map[string]string, len(pm.File.Require))
		for i, req := range pm.File.Require {
			why[req.Mod.Path] = whyList[i]
		}
		return &modWhyData{why: why}
	}, nil)

	mwh := &modWhyHandle{handle: h}
	s.mu.Lock()
	s.modWhyHandles[fh.URI()] = mwh
	s.mu.Unlock()

	return mwh.why(ctx, s)
}

// extractGoCommandError tries to parse errors that come from the go command
// and shape them into go.mod diagnostics.
func (s *snapshot) extractGoCommandErrors(ctx context.Context, goCmdError string) ([]*source.Diagnostic, error) {
	diagLocations := map[*source.ParsedModule]span.Span{}
	backupDiagLocations := map[*source.ParsedModule]span.Span{}

	// The go command emits parse errors for completely invalid go.mod files.
	// Those are reported by our own diagnostics and can be ignored here.
	// As of writing, we are not aware of any other errors that include
	// file/position information, so don't even try to find it.
	if strings.Contains(goCmdError, "errors parsing go.mod") {
		return nil, nil
	}

	// Match the error against all the mod files in the workspace.
	for _, uri := range s.ModFiles() {
		fh, err := s.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		pm, err := s.ParseMod(ctx, fh)
		if err != nil {
			return nil, err
		}
		spn, found, err := s.matchErrorToModule(ctx, pm, goCmdError)
		if err != nil {
			return nil, err
		}
		if found {
			diagLocations[pm] = spn
		} else {
			backupDiagLocations[pm] = spn
		}
	}

	// If we didn't find any good matches, assign diagnostics to all go.mod files.
	if len(diagLocations) == 0 {
		diagLocations = backupDiagLocations
	}

	var srcErrs []*source.Diagnostic
	for pm, spn := range diagLocations {
		diag, err := s.goCommandDiagnostic(pm, spn, goCmdError)
		if err != nil {
			return nil, err
		}
		srcErrs = append(srcErrs, diag)
	}
	return srcErrs, nil
}

var moduleVersionInErrorRe = regexp.MustCompile(`[:\s]([+-._~0-9A-Za-z]+)@([+-._~0-9A-Za-z]+)[:\s]`)

// matchErrorToModule matches a go command error message to a go.mod file.
// Some examples:
//
//    example.com@v1.2.2: reading example.com/@v/v1.2.2.mod: no such file or directory
//    go: github.com/cockroachdb/apd/v2@v2.0.72: reading github.com/cockroachdb/apd/go.mod at revision v2.0.72: unknown revision v2.0.72
//    go: example.com@v1.2.3 requires\n\trandom.org@v1.2.3: parsing go.mod:\n\tmodule declares its path as: bob.org\n\tbut was required as: random.org
//
// It returns the location of a reference to the one of the modules and true
// if one exists. If none is found it returns a fallback location and false.
func (s *snapshot) matchErrorToModule(ctx context.Context, pm *source.ParsedModule, goCmdError string) (span.Span, bool, error) {
	var reference *modfile.Line
	matches := moduleVersionInErrorRe.FindAllStringSubmatch(goCmdError, -1)

	for i := len(matches) - 1; i >= 0; i-- {
		ver := module.Version{Path: matches[i][1], Version: matches[i][2]}
		// Any module versions that come from the workspace module should not
		// be shown to the user.
		if source.IsWorkspaceModuleVersion(ver.Version) {
			continue
		}
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
		if pm.File.Module == nil {
			return span.New(pm.URI, span.NewPoint(1, 1, 0), span.Point{}), false, nil
		}
		spn, err := spanFromPositions(pm.Mapper, pm.File.Module.Syntax.Start, pm.File.Module.Syntax.End)
		return spn, false, err
	}

	spn, err := spanFromPositions(pm.Mapper, reference.Start, reference.End)
	return spn, true, err
}

// goCommandDiagnostic creates a diagnostic for a given go command error.
func (s *snapshot) goCommandDiagnostic(pm *source.ParsedModule, spn span.Span, goCmdError string) (*source.Diagnostic, error) {
	rng, err := pm.Mapper.Range(spn)
	if err != nil {
		return nil, err
	}

	matches := moduleVersionInErrorRe.FindAllStringSubmatch(goCmdError, -1)
	var innermost *module.Version
	for i := len(matches) - 1; i >= 0; i-- {
		ver := module.Version{Path: matches[i][1], Version: matches[i][2]}
		// Any module versions that come from the workspace module should not
		// be shown to the user.
		if source.IsWorkspaceModuleVersion(ver.Version) {
			continue
		}
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
			Range:    rng,
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
			Range:    rng,
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
			Range:          rng,
			Severity:       protocol.SeverityError,
			Message:        fmt.Sprintf("%v@%v has not been downloaded", innermost.Path, innermost.Version),
			Source:         source.ListError,
			SuggestedFixes: []source.SuggestedFix{source.SuggestedFixFromCommand(cmd, protocol.QuickFix)},
		}, nil
	default:
		return &source.Diagnostic{
			URI:      pm.URI,
			Range:    rng,
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
