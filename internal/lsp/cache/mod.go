// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

const (
	SyntaxError = "syntax"
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
		var parseErrors []source.Error
		if err != nil {
			if parseErr, extractErr := extractModParseErrors(modFH.URI(), m, err, contents); extractErr == nil {
				parseErrors = []source.Error{*parseErr}
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

// extractModParseErrors processes the raw errors returned by modfile.Parse,
// extracting the filenames and line numbers that correspond to the errors.
func extractModParseErrors(uri span.URI, m *protocol.ColumnMapper, parseErr error, content []byte) (*source.Error, error) {
	re := regexp.MustCompile(`.*:([\d]+): (.+)`)
	matches := re.FindStringSubmatch(strings.TrimSpace(parseErr.Error()))
	if len(matches) < 3 {
		return nil, errors.Errorf("could not parse go.mod error message: %s", parseErr)
	}
	line, err := strconv.Atoi(matches[1])
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(content), "\n")
	if line > len(lines) {
		return nil, errors.Errorf("could not parse go.mod error message %q, line number %v out of range", content, line)
	}
	// The error returned from the modfile package only returns a line number,
	// so we assume that the diagnostic should be for the entire line.
	endOfLine := len(lines[line-1])
	sOffset, err := m.Converter.ToOffset(line, 0)
	if err != nil {
		return nil, err
	}
	eOffset, err := m.Converter.ToOffset(line, endOfLine)
	if err != nil {
		return nil, err
	}
	spn := span.New(uri, span.NewPoint(line, 0, sOffset), span.NewPoint(line, endOfLine, eOffset))
	rng, err := m.Range(spn)
	if err != nil {
		return nil, err
	}
	return &source.Error{
		Category: SyntaxError,
		Message:  matches[2],
		Range:    rng,
		URI:      uri,
	}, nil
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

type modUpgradeHandle struct {
	handle *memoize.Handle
}

type modUpgradeData struct {
	// upgrades maps modules to their latest versions.
	upgrades map[string]string

	err error
}

func (muh *modUpgradeHandle) upgrades(ctx context.Context, snapshot *snapshot) (map[string]string, error) {
	v, err := muh.handle.Get(ctx, snapshot.generation, snapshot)
	if v == nil {
		return nil, err
	}
	data := v.(*modUpgradeData)
	return data.upgrades, data.err
}

// moduleUpgrade describes a module that can be upgraded to a particular
// version.
type moduleUpgrade struct {
	Path   string
	Update struct {
		Version string
	}
}

func (s *snapshot) ModUpgrade(ctx context.Context, fh source.FileHandle) (map[string]string, error) {
	if fh.Kind() != source.Mod {
		return nil, fmt.Errorf("%s is not a go.mod file", fh.URI())
	}
	if handle := s.getModUpgradeHandle(fh.URI()); handle != nil {
		return handle.upgrades(ctx, s)
	}
	key := modKey{
		sessionID: s.view.session.id,
		env:       hashEnv(s),
		mod:       fh.FileIdentity(),
		view:      s.view.rootURI.Filename(),
		verb:      upgrade,
	}
	h := s.generation.Bind(key, func(ctx context.Context, arg memoize.Arg) interface{} {
		ctx, done := event.Start(ctx, "cache.ModUpgradeHandle", tag.URI.Of(fh.URI()))
		defer done()

		snapshot := arg.(*snapshot)

		pm, err := snapshot.ParseMod(ctx, fh)
		if err != nil {
			return &modUpgradeData{err: err}
		}

		// No requires to upgrade.
		if len(pm.File.Require) == 0 {
			return &modUpgradeData{}
		}
		// Run "go list -mod readonly -u -m all" to be able to see which deps can be
		// upgraded without modifying mod file.
		inv := &gocommand.Invocation{
			Verb:       "list",
			Args:       []string{"-u", "-m", "-json", "all"},
			WorkingDir: filepath.Dir(fh.URI().Filename()),
		}
		if s.workspaceMode()&tempModfile == 0 || containsVendor(fh.URI()) {
			// Use -mod=readonly if the module contains a vendor directory
			// (see golang/go#38711).
			inv.ModFlag = "readonly"
		}
		stdout, err := snapshot.RunGoCommandDirect(ctx, source.Normal, inv)
		if err != nil {
			return &modUpgradeData{err: err}
		}
		var upgradeList []moduleUpgrade
		dec := json.NewDecoder(stdout)
		for {
			var m moduleUpgrade
			if err := dec.Decode(&m); err == io.EOF {
				break
			} else if err != nil {
				return &modUpgradeData{err: err}
			}
			upgradeList = append(upgradeList, m)
		}
		if len(upgradeList) <= 1 {
			return &modUpgradeData{}
		}
		upgrades := make(map[string]string)
		for _, upgrade := range upgradeList[1:] {
			if upgrade.Update.Version == "" {
				continue
			}
			upgrades[upgrade.Path] = upgrade.Update.Version
		}
		return &modUpgradeData{
			upgrades: upgrades,
		}
	}, nil)
	muh := &modUpgradeHandle{handle: h}
	s.mu.Lock()
	s.modUpgradeHandles[fh.URI()] = muh
	s.mu.Unlock()

	return muh.upgrades(ctx, s)
}

// containsVendor reports whether the module has a vendor folder.
func containsVendor(modURI span.URI) bool {
	dir := filepath.Dir(modURI.Filename())
	f, err := os.Stat(filepath.Join(dir, "vendor"))
	if err != nil {
		return false
	}
	return f.IsDir()
}

var moduleAtVersionRe = regexp.MustCompile(`^(?P<module>.*)@(?P<version>.*)$`)

// ExtractGoCommandError tries to parse errors that come from the go command
// and shape them into go.mod diagnostics.
func extractGoCommandError(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle, loadErr error) *source.Error {
	// We try to match module versions in error messages. Some examples:
	//
	//  example.com@v1.2.2: reading example.com/@v/v1.2.2.mod: no such file or directory
	//  go: github.com/cockroachdb/apd/v2@v2.0.72: reading github.com/cockroachdb/apd/go.mod at revision v2.0.72: unknown revision v2.0.72
	//  go: example.com@v1.2.3 requires\n\trandom.org@v1.2.3: parsing go.mod:\n\tmodule declares its path as: bob.org\n\tbut was required as: random.org
	//
	// We split on colons and whitespace, and attempt to match on something
	// that matches module@version. If we're able to find a match, we try to
	// find anything that matches it in the go.mod file.
	var v module.Version
	fields := strings.FieldsFunc(loadErr.Error(), func(r rune) bool {
		return unicode.IsSpace(r) || r == ':'
	})
	for _, s := range fields {
		s = strings.TrimSpace(s)
		match := moduleAtVersionRe.FindStringSubmatch(s)
		if match == nil || len(match) < 3 {
			continue
		}
		path, version := match[1], match[2]
		// Any module versions that come from the workspace module should not
		// be shown to the user.
		if source.IsWorkspaceModuleVersion(version) {
			continue
		}
		if err := module.Check(path, version); err != nil {
			continue
		}
		v.Path, v.Version = path, version
		break
	}
	pm, err := snapshot.ParseMod(ctx, fh)
	if err != nil {
		return nil
	}
	toSourceError := func(line *modfile.Line) *source.Error {
		rng, err := rangeFromPositions(pm.Mapper, line.Start, line.End)
		if err != nil {
			return nil
		}
		return &source.Error{
			Message: loadErr.Error(),
			Range:   rng,
			URI:     fh.URI(),
		}
	}
	// Check if there are any require, exclude, or replace statements that
	// match this module version.
	for _, req := range pm.File.Require {
		if req.Mod != v {
			continue
		}
		return toSourceError(req.Syntax)
	}
	for _, ex := range pm.File.Exclude {
		if ex.Mod != v {
			continue
		}
		return toSourceError(ex.Syntax)
	}
	for _, rep := range pm.File.Replace {
		if rep.New != v && rep.Old != v {
			continue
		}
		return toSourceError(rep.Syntax)
	}
	// No match for the module path was found in the go.mod file.
	// Show the error on the module declaration, if one exists.
	if pm.File.Module == nil {
		return nil
	}
	return toSourceError(pm.File.Module.Syntax)
}
