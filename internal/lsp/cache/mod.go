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

	mod, sum source.FileHandle
}

type parseModData struct {
	memoize.NoCopy

	parsed *modfile.File
	m      *protocol.ColumnMapper

	// parseErrors refers to syntax errors found in the go.mod file.
	parseErrors []source.Error

	// err is any error encountered while parsing the file.
	err error
}

func (mh *parseModHandle) Mod() source.FileHandle {
	return mh.mod
}

func (mh *parseModHandle) Sum() source.FileHandle {
	return mh.sum
}

func (mh *parseModHandle) Parse(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, []source.Error, error) {
	v, err := mh.handle.Get(ctx)
	if err != nil {
		return nil, nil, nil, err
	}
	data := v.(*parseModData)
	return data.parsed, data.m, data.parseErrors, data.err
}

func (s *snapshot) ParseModHandle(ctx context.Context, modFH source.FileHandle) (source.ParseModHandle, error) {
	if handle := s.getModHandle(modFH.URI()); handle != nil {
		return handle, nil
	}
	h := s.view.session.cache.store.Bind(modFH.Identity().String(), func(ctx context.Context) interface{} {
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
		parsed, err := modfile.Parse(modFH.URI().Filename(), contents, nil)
		if err != nil {
			parseErr, _ := extractModParseErrors(modFH.URI(), m, err, contents)
			var parseErrors []source.Error
			if parseErr != nil {
				parseErrors = append(parseErrors, *parseErr)
			}
			return &parseModData{
				parseErrors: parseErrors,
				err:         err,
			}
		}
		return &parseModData{
			parsed: parsed,
			m:      m,
		}
	})
	// Get the go.sum file, either from the snapshot or directly from the
	// cache. Avoid (*snapshot).GetFile here, as we don't want to add
	// nonexistent file handles to the snapshot if the file does not exist.
	sumURI := span.URIFromPath(sumFilename(modFH.URI()))
	sumFH := s.FindFile(sumURI)
	if sumFH == nil {
		fh, err := s.view.session.cache.getFile(ctx, sumURI)
		if err != nil && !os.IsNotExist(err) {
			return nil, err
		}
		if fh.err != nil && !os.IsNotExist(fh.err) {
			return nil, fh.err
		}
		// If the file doesn't exist, we can just keep the go.sum nil.
		if err != nil || fh.err != nil {
			sumFH = nil
		} else {
			sumFH = fh
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.parseModHandles[modFH.URI()] = &parseModHandle{
		handle: h,
		mod:    modFH,
		sum:    sumFH,
	}
	return s.parseModHandles[modFH.URI()], nil
}

func sumFilename(modURI span.URI) string {
	return modURI.Filename()[:len(modURI.Filename())-len("mod")] + "sum"
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
	sessionID, cfg, mod, view string
	verb                      modAction
}

type modAction int

const (
	why modAction = iota
	upgrade
)

type modWhyHandle struct {
	handle *memoize.Handle

	pmh source.ParseModHandle
}

type modWhyData struct {
	// why keeps track of the `go mod why` results for each require statement
	// in the go.mod file.
	why map[string]string

	err error
}

func (mwh *modWhyHandle) Why(ctx context.Context) (map[string]string, error) {
	v, err := mwh.handle.Get(ctx)
	if err != nil {
		return nil, err
	}
	data := v.(*modWhyData)
	return data.why, data.err
}

func (s *snapshot) ModWhyHandle(ctx context.Context) (source.ModWhyHandle, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	fh, err := s.GetFile(ctx, s.view.modURI)
	if err != nil {
		return nil, err
	}
	pmh, err := s.ParseModHandle(ctx, fh)
	if err != nil {
		return nil, err
	}
	var (
		cfg    = s.config(ctx)
		tmpMod = s.view.tmpMod
	)
	key := modKey{
		sessionID: s.view.session.id,
		cfg:       hashConfig(cfg),
		mod:       pmh.Mod().Identity().String(),
		view:      s.view.root.Filename(),
		verb:      why,
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		ctx, done := event.Start(ctx, "cache.ModHandle", tag.URI.Of(pmh.Mod().URI()))
		defer done()

		parsed, _, _, err := pmh.Parse(ctx)
		if err != nil {
			return &modWhyData{err: err}
		}
		// No requires to explain.
		if len(parsed.Require) == 0 {
			return &modWhyData{}
		}
		// Run `go mod why` on all the dependencies.
		args := []string{"why", "-m"}
		for _, req := range parsed.Require {
			args = append(args, req.Mod.Path)
		}
		_, stdout, err := runGoCommand(ctx, cfg, pmh, tmpMod, "mod", args)
		if err != nil {
			return &modWhyData{err: err}
		}
		whyList := strings.Split(stdout.String(), "\n\n")
		if len(whyList) != len(parsed.Require) {
			return &modWhyData{
				err: fmt.Errorf("mismatched number of results: got %v, want %v", len(whyList), len(parsed.Require)),
			}
		}
		why := make(map[string]string, len(parsed.Require))
		for i, req := range parsed.Require {
			why[req.Mod.Path] = whyList[i]
		}
		return &modWhyData{why: why}
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modWhyHandle = &modWhyHandle{
		handle: h,
		pmh:    pmh,
	}
	return s.modWhyHandle, nil
}

type modUpgradeHandle struct {
	handle *memoize.Handle

	pmh source.ParseModHandle
}

type modUpgradeData struct {
	// upgrades maps modules to their latest versions.
	upgrades map[string]string

	err error
}

func (muh *modUpgradeHandle) Upgrades(ctx context.Context) (map[string]string, error) {
	v, err := muh.handle.Get(ctx)
	if v == nil {
		return nil, err
	}
	data := v.(*modUpgradeData)
	return data.upgrades, data.err
}

func (s *snapshot) ModUpgradeHandle(ctx context.Context) (source.ModUpgradeHandle, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	fh, err := s.GetFile(ctx, s.view.modURI)
	if err != nil {
		return nil, err
	}
	pmh, err := s.ParseModHandle(ctx, fh)
	if err != nil {
		return nil, err
	}
	var (
		cfg    = s.config(ctx)
		tmpMod = s.view.tmpMod
	)
	key := modKey{
		sessionID: s.view.session.id,
		cfg:       hashConfig(cfg),
		mod:       pmh.Mod().Identity().String(),
		view:      s.view.root.Filename(),
		verb:      upgrade,
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		ctx, done := event.Start(ctx, "cache.ModUpgradeHandle", tag.URI.Of(pmh.Mod().URI()))
		defer done()

		parsed, _, _, err := pmh.Parse(ctx)
		if err != nil {
			return &modUpgradeData{err: err}
		}
		// No requires to upgrade.
		if len(parsed.Require) == 0 {
			return &modUpgradeData{}
		}
		// Run "go list -mod readonly -u -m all" to be able to see which deps can be
		// upgraded without modifying mod file.
		args := []string{"-u", "-m", "all"}
		if !tmpMod || containsVendor(pmh.Mod().URI()) {
			// Use -mod=readonly if the module contains a vendor directory
			// (see golang/go#38711).
			args = append([]string{"-mod", "readonly"}, args...)
		}
		_, stdout, err := runGoCommand(ctx, cfg, pmh, tmpMod, "list", args)
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
			if len(info) < 3 {
				continue
			}
			dep, version := info[0], info[2]
			latest := version[1:]                    // remove the "["
			latest = strings.TrimSuffix(latest, "]") // remove the "]"
			upgrades[dep] = latest
		}
		return &modUpgradeData{
			upgrades: upgrades,
		}
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modUpgradeHandle = &modUpgradeHandle{
		handle: h,
		pmh:    pmh,
	}
	return s.modUpgradeHandle, nil
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
