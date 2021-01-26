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
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
)

const (
	SyntaxError    = "syntax"
	GoCommandError = "go command"
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
		var parseErrors []*source.Error
		if err != nil {
			if parseErr := extractErrorWithPosition(ctx, err.Error(), s); parseErr != nil {
				parseErrors = []*source.Error{parseErr}
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
func (s *snapshot) extractGoCommandErrors(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, goCmdError string) []*source.Error {
	var srcErrs []*source.Error
	if srcErr := s.parseModError(ctx, fh, goCmdError); srcErr != nil {
		srcErrs = append(srcErrs, srcErr)
	}
	// If the error message contains a position, use that. Don't pass a file
	// handle in, as it might not be the file associated with the error.
	if srcErr := extractErrorWithPosition(ctx, goCmdError, s); srcErr != nil {
		srcErrs = append(srcErrs, srcErr)
	} else if srcErr := s.matchErrorToModule(ctx, fh, goCmdError); srcErr != nil {
		srcErrs = append(srcErrs, srcErr)
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
func (s *snapshot) matchErrorToModule(ctx context.Context, fh source.FileHandle, goCmdError string) *source.Error {
	pm, err := s.ParseMod(ctx, fh)
	if err != nil {
		return nil
	}

	var innermost *module.Version
	var reference *modfile.Line
	matches := moduleVersionInErrorRe.FindAllStringSubmatch(goCmdError, -1)

outer:
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

		for _, req := range pm.File.Require {
			if req.Mod == ver {
				reference = req.Syntax
				break outer
			}
		}
		for _, ex := range pm.File.Exclude {
			if ex.Mod == ver {
				reference = ex.Syntax
				break outer
			}
		}
		for _, rep := range pm.File.Replace {
			if rep.New == ver || rep.Old == ver {
				reference = rep.Syntax
				break outer
			}
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
		args, err := source.MarshalArgs(fh.URI(), false, []string{fmt.Sprintf("%v@%v", innermost.Path, innermost.Version)})
		if err != nil {
			return nil
		}
		msg := goCmdError
		if disabledByGOPROXY {
			msg = fmt.Sprintf("%v@%v has not been downloaded", innermost.Path, innermost.Version)
		}
		return &source.Error{
			Message: msg,
			Kind:    source.ListError,
			Range:   rng,
			URI:     fh.URI(),
			SuggestedFixes: []source.SuggestedFix{{
				Title: fmt.Sprintf("Download %v@%v", innermost.Path, innermost.Version),
				Command: &protocol.Command{
					Title:     source.CommandAddDependency.Title,
					Command:   source.CommandAddDependency.ID(),
					Arguments: args,
				},
			}},
		}
	}
	return &source.Error{
		Message: goCmdError,
		Range:   rng,
		URI:     fh.URI(),
		Kind:    source.ListError,
	}
}

// errorPositionRe matches errors messages of the form <filename>:<line>:<col>,
// where the <col> is optional.
var errorPositionRe = regexp.MustCompile(`(?P<pos>.*:([\d]+)(:([\d]+))?): (?P<msg>.+)`)

// extractErrorWithPosition returns a structured error with position
// information for the given unstructured error. If a file handle is provided,
// the error position will be on that file. This is useful for parse errors,
// where we already know the file with the error.
func extractErrorWithPosition(ctx context.Context, goCmdError string, src source.FileSource) *source.Error {
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
	category := GoCommandError
	if fh != nil {
		category = SyntaxError
	}
	return &source.Error{
		Category: category,
		Message:  msg,
		Range:    rng,
		URI:      spn.URI(),
	}
}
