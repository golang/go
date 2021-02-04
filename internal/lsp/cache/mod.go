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
		file, err := modfile.Parse(modFH.URI().Filename(), contents, nil)

		// Attempt to convert the error to a standardized parse error.
		var parseErrors []*source.Diagnostic
		if err != nil {
			if parseErr := extractErrorWithPosition(ctx, err.Error(), s); parseErr != nil {
				parseErrors = []*source.Diagnostic{parseErr}
			}
		}
		return &parseModData{
			parsed: &source.ParsedModule{
				URI:         modFH.URI(),
				Mapper:      m,
				File:        file,
				ParseErrors: parseErrors,
			},
			err: err,
		}
	}, nil)

	pmh := &parseModHandle{handle: h}
	s.mu.Lock()
	s.parseModHandles[modFH.URI()] = pmh
	s.mu.Unlock()

	return pmh.parse(ctx, s)
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
	if fh.Kind() != source.Mod {
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
func (s *snapshot) extractGoCommandErrors(ctx context.Context, snapshot source.Snapshot, goCmdError string) []*source.Diagnostic {
	var srcErrs []*source.Diagnostic
	if srcErr := s.parseModError(ctx, goCmdError); srcErr != nil {
		srcErrs = append(srcErrs, srcErr...)
	}
	if srcErr := extractErrorWithPosition(ctx, goCmdError, s); srcErr != nil {
		srcErrs = append(srcErrs, srcErr)
	} else {
		for _, uri := range s.ModFiles() {
			fh, err := s.GetFile(ctx, uri)
			if err != nil {
				continue
			}
			if srcErr := s.matchErrorToModule(ctx, fh, goCmdError); srcErr != nil {
				srcErrs = append(srcErrs, srcErr)
			}
		}
	}
	return srcErrs
}

var moduleVersionInErrorRe = regexp.MustCompile(`[:\s]([+-._~0-9A-Za-z]+)@([+-._~0-9A-Za-z]+)[:\s]`)

// matchErrorToModule attempts to match module version in error messages.
// Some examples:
//
//    example.com@v1.2.2: reading example.com/@v/v1.2.2.mod: no such file or directory
//    go: github.com/cockroachdb/apd/v2@v2.0.72: reading github.com/cockroachdb/apd/go.mod at revision v2.0.72: unknown revision v2.0.72
//    go: example.com@v1.2.3 requires\n\trandom.org@v1.2.3: parsing go.mod:\n\tmodule declares its path as: bob.org\n\tbut was required as: random.org
//
// We search for module@version, starting from the end to find the most
// relevant module, e.g. random.org@v1.2.3 above. Then we associate the error
// with a directive that references any of the modules mentioned.
func (s *snapshot) matchErrorToModule(ctx context.Context, fh source.FileHandle, goCmdError string) *source.Diagnostic {
	pm, err := s.ParseMod(ctx, fh)
	if err != nil {
		return nil
	}

	var innermost *module.Version
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
		if innermost == nil {
			innermost = &ver
		}
		reference = findModuleReference(pm.File, ver)
		if reference != nil {
			break
		}
	}

	if reference == nil {
		// No match for the module path was found in the go.mod file.
		// Show the error on the module declaration, if one exists.
		if pm.File.Module == nil {
			return nil
		}
		reference = pm.File.Module.Syntax
	}

	rng, err := rangeFromPositions(pm.Mapper, reference.Start, reference.End)
	if err != nil {
		return nil
	}
	disabledByGOPROXY := strings.Contains(goCmdError, "disabled by GOPROXY=off")
	shouldAddDep := strings.Contains(goCmdError, "to add it")
	if innermost != nil && (disabledByGOPROXY || shouldAddDep) {
		title := fmt.Sprintf("Download %v@%v", innermost.Path, innermost.Version)
		cmd, err := command.NewAddDependencyCommand(title, command.DependencyArgs{
			URI:        protocol.URIFromSpanURI(fh.URI()),
			AddRequire: false,
			GoCmdArgs:  []string{fmt.Sprintf("%v@%v", innermost.Path, innermost.Version)},
		})
		if err != nil {
			return nil
		}
		msg := goCmdError
		if disabledByGOPROXY {
			msg = fmt.Sprintf("%v@%v has not been downloaded", innermost.Path, innermost.Version)
		}
		return &source.Diagnostic{
			URI:            fh.URI(),
			Range:          rng,
			Severity:       protocol.SeverityError,
			Message:        msg,
			Source:         source.ListError,
			SuggestedFixes: []source.SuggestedFix{source.SuggestedFixFromCommand(cmd)},
		}
	}
	diagSource := source.ListError
	if fh != nil {
		diagSource = source.ParseError
	}
	return &source.Diagnostic{
		URI:      fh.URI(),
		Range:    rng,
		Severity: protocol.SeverityError,
		Source:   diagSource,
		Message:  goCmdError,
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

// errorPositionRe matches errors messages of the form <filename>:<line>:<col>,
// where the <col> is optional.
var errorPositionRe = regexp.MustCompile(`(?P<pos>.*:([\d]+)(:([\d]+))?): (?P<msg>.+)`)

// extractErrorWithPosition returns a structured error with position
// information for the given unstructured error. If a file handle is provided,
// the error position will be on that file. This is useful for parse errors,
// where we already know the file with the error.
func extractErrorWithPosition(ctx context.Context, goCmdError string, src source.FileSource) *source.Diagnostic {
	matches := errorPositionRe.FindStringSubmatch(strings.TrimSpace(goCmdError))
	if len(matches) == 0 {
		return nil
	}
	var pos, msg string
	for i, name := range errorPositionRe.SubexpNames() {
		if name == "pos" {
			pos = matches[i]
		}
		if name == "msg" {
			msg = matches[i]
		}
	}
	spn := span.Parse(pos)
	fh, err := src.GetFile(ctx, spn.URI())
	if err != nil {
		return nil
	}
	content, err := fh.Read()
	if err != nil {
		return nil
	}
	m := &protocol.ColumnMapper{
		URI:       spn.URI(),
		Converter: span.NewContentConverter(spn.URI().Filename(), content),
		Content:   content,
	}
	rng, err := m.Range(spn)
	if err != nil {
		return nil
	}
	diagSource := source.ListError
	if fh != nil {
		diagSource = source.ParseError
	}
	return &source.Diagnostic{
		URI:      spn.URI(),
		Range:    rng,
		Severity: protocol.SeverityError,
		Source:   diagSource,
		Message:  msg,
	}
}
