// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
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
		data := &parseModData{
			parsed: &source.ParsedModule{
				Mapper: m,
			},
		}
		data.parsed.File, data.err = modfile.Parse(modFH.URI().Filename(), contents, nil)
		if data.err != nil {
			// Attempt to convert the error to a standardized parse error.
			if parseErr, extractErr := extractModParseErrors(modFH.URI(), m, data.err, contents); extractErr == nil {
				data.parsed.ParseErrors = []source.Error{*parseErr}
			}
			// If the file was still parsed, we don't want to treat this as a
			// fatal error. Note: This currently cannot happen as modfile.Parse
			// always returns an error when the file is nil.
			if data.parsed.File != nil {
				data.err = nil
			}
		}
		return data
	})

	pmh := &parseModHandle{handle: h}
	s.mu.Lock()
	s.parseModHandles[modFH.URI()] = pmh
	s.mu.Unlock()

	return pmh.parse(ctx, s)
}

func (s *snapshot) sumFH(ctx context.Context, modFH source.FileHandle) (source.FileHandle, error) {
	// Get the go.sum file, either from the snapshot or directly from the
	// cache. Avoid (*snapshot).GetFile here, as we don't want to add
	// nonexistent file handles to the snapshot if the file does not exist.
	sumURI := span.URIFromPath(sumFilename(modFH.URI()))
	var sumFH source.FileHandle = s.FindFile(sumURI)
	if sumFH == nil {
		var err error
		sumFH, err = s.view.session.cache.getFile(ctx, sumURI)
		if err != nil {
			return nil, err
		}
	}
	_, err := sumFH.Read()
	if err != nil {
		return nil, err
	}
	return sumFH, nil
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
	sessionID, cfg, view string
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
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	if handle := s.getModWhyHandle(fh.URI()); handle != nil {
		return handle.why(ctx, s)
	}
	// Make sure to use the module root as the working directory.
	cfg := s.configWithDir(ctx, filepath.Dir(fh.URI().Filename()))
	key := modKey{
		sessionID: s.view.session.id,
		cfg:       hashConfig(cfg),
		mod:       fh.FileIdentity(),
		view:      s.view.root.Filename(),
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
		args := []string{"why", "-m"}
		for _, req := range pm.File.Require {
			args = append(args, req.Mod.Path)
		}
		stdout, err := snapshot.RunGoCommand(ctx, "mod", args)
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
	})

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

func (s *snapshot) ModUpgrade(ctx context.Context, fh source.FileHandle) (map[string]string, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	if handle := s.getModUpgradeHandle(fh.URI()); handle != nil {
		return handle.upgrades(ctx, s)
	}
	// Use the module root as the working directory.
	cfg := s.configWithDir(ctx, filepath.Dir(fh.URI().Filename()))
	key := modKey{
		sessionID: s.view.session.id,
		cfg:       hashConfig(cfg),
		mod:       fh.FileIdentity(),
		view:      s.view.root.Filename(),
		verb:      upgrade,
	}
	h := s.generation.Bind(key, func(ctx context.Context, arg memoize.Arg) interface{} {
		ctx, done := event.Start(ctx, "cache.ModUpgradeHandle", tag.URI.Of(fh.URI()))
		defer done()

		snapshot := arg.(*snapshot)

		pm, err := s.ParseMod(ctx, fh)
		if err != nil {
			return &modUpgradeData{err: err}
		}

		// No requires to upgrade.
		if len(pm.File.Require) == 0 {
			return &modUpgradeData{}
		}
		// Run "go list -mod readonly -u -m all" to be able to see which deps can be
		// upgraded without modifying mod file.
		args := []string{"-u", "-m", "all"}
		if !snapshot.view.tmpMod || containsVendor(fh.URI()) {
			// Use -mod=readonly if the module contains a vendor directory
			// (see golang/go#38711).
			args = append([]string{"-mod", "readonly"}, args...)
		}
		stdout, err := snapshot.RunGoCommand(ctx, "list", args)
		if err != nil {
			return &modUpgradeData{err: err}
		}
		upgradesList := strings.Split(stdout.String(), "\n")
		if len(upgradesList) <= 1 {
			return nil
		}
		upgrades := make(map[string]string)
		for _, upgrade := range upgradesList[1:] {
			// Example: "github.com/x/tools v1.1.0 [v1.2.0]"
			info := strings.Split(upgrade, " ")
			if len(info) != 3 {
				continue
			}
			dep, version := info[0], info[2]

			// Make sure that the format matches our expectation.
			if len(version) < 2 {
				continue
			}
			if version[0] != '[' || version[len(version)-1] != ']' {
				continue
			}
			latest := version[1 : len(version)-1] // remove the "[" and "]"
			upgrades[dep] = latest
		}
		return &modUpgradeData{
			upgrades: upgrades,
		}
	})
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
