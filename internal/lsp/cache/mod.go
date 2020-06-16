// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

const (
	ModTidyError = "go mod tidy"
	SyntaxError  = "syntax"
)

type modKey struct {
	sessionID string
	cfg       string
	gomod     string
	view      string
}

type modTidyKey struct {
	sessionID       string
	cfg             string
	gomod           string
	imports         string
	unsavedOverlays string
	view            string
}

type modHandle struct {
	handle *memoize.Handle

	file source.FileHandle
	cfg  *packages.Config
}

type modData struct {
	memoize.NoCopy

	// parsed contains the parsed contents that are used to diff with
	// the ideal contents.
	parsed *modfile.File

	// m is the column mapper for the original go.mod file.
	m *protocol.ColumnMapper

	// upgrades is a map of path->version that contains any upgrades for the go.mod.
	upgrades map[string]string

	// why is a map of path->explanation that contains all the "go mod why" contents
	// for each require statement.
	why map[string]string

	// err is any error that occurs while we are calculating the parseErrors.
	err error
}

func (mh *modHandle) String() string {
	return mh.File().URI().Filename()
}

func (mh *modHandle) File() source.FileHandle {
	return mh.file
}

func (mh *modHandle) Parse(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, error) {
	v := mh.handle.Get(ctx)
	if v == nil {
		return nil, nil, errors.Errorf("no parsed file for %s", mh.File().URI())
	}
	data := v.(*modData)
	return data.parsed, data.m, data.err
}

func (mh *modHandle) Upgrades(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, map[string]string, error) {
	v := mh.handle.Get(ctx)
	if v == nil {
		return nil, nil, nil, errors.Errorf("no parsed file for %s", mh.File().URI())
	}
	data := v.(*modData)
	return data.parsed, data.m, data.upgrades, data.err
}

func (mh *modHandle) Why(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, map[string]string, error) {
	v := mh.handle.Get(ctx)
	if v == nil {
		return nil, nil, nil, errors.Errorf("no parsed file for %s", mh.File().URI())
	}
	data := v.(*modData)
	return data.parsed, data.m, data.why, data.err
}

func (s *snapshot) ModHandle(ctx context.Context, modFH source.FileHandle) (source.ModHandle, error) {
	if err := s.awaitLoaded(ctx); err != nil {
		return nil, err
	}
	var sumFH source.FileHandle
	if s.view.sumURI != "" {
		var err error
		sumFH, err = s.GetFile(ctx, s.view.sumURI)
		if err != nil {
			return nil, err
		}
	}
	var (
		cfg    = s.config(ctx)
		modURI = s.view.modURI
		tmpMod = s.view.tmpMod
	)
	if handle := s.getModHandle(modFH.URI()); handle != nil {
		return handle, nil
	}
	key := modKey{
		sessionID: s.view.session.id,
		cfg:       hashConfig(cfg),
		gomod:     modFH.Identity().String(),
		view:      s.view.folder.Filename(),
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		ctx, done := event.Start(ctx, "cache.ModHandle", tag.URI.Of(modFH.URI()))
		defer done()

		contents, err := modFH.Read()
		if err != nil {
			return &modData{
				err: err,
			}
		}
		parsedFile, err := modfile.Parse(modFH.URI().Filename(), contents, nil)
		if err != nil {
			return &modData{
				err: err,
			}
		}
		data := &modData{
			parsed: parsedFile,
			m: &protocol.ColumnMapper{
				URI:       modFH.URI(),
				Converter: span.NewContentConverter(modFH.URI().Filename(), contents),
				Content:   contents,
			},
		}

		// If this go.mod file is not the view's go.mod file, or if the
		// -modfile flag is not supported, then we just want to parse.
		if modFH.URI() != modURI {
			return data
		}

		// Only get dependency upgrades if the go.mod file is the same as the view's.
		if err := dependencyUpgrades(ctx, cfg, modFH, sumFH, tmpMod, data); err != nil {
			return &modData{err: err}
		}
		// Only run "go mod why" if the go.mod file is the same as the view's.
		if err := goModWhy(ctx, cfg, modFH, sumFH, tmpMod, data); err != nil {
			return &modData{err: err}
		}
		return data
	})
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modHandles[modFH.URI()] = &modHandle{
		handle: h,
		file:   modFH,
		cfg:    cfg,
	}
	return s.modHandles[modFH.URI()], nil
}

func goModWhy(ctx context.Context, cfg *packages.Config, modFH, sumFH source.FileHandle, tmpMod bool, data *modData) error {
	if len(data.parsed.Require) == 0 {
		return nil
	}
	// Run "go mod why" on all the dependencies.
	args := []string{"why", "-m"}
	for _, req := range data.parsed.Require {
		args = append(args, req.Mod.Path)
	}
	// If the -modfile flag is disabled, don't pass in a go.mod URI.
	if !tmpMod {
		modFH = nil
	}
	_, stdout, err := runGoCommand(ctx, cfg, modFH, sumFH, "mod", args)
	if err != nil {
		return err
	}
	whyList := strings.Split(stdout.String(), "\n\n")
	if len(whyList) != len(data.parsed.Require) {
		return nil
	}
	data.why = make(map[string]string, len(data.parsed.Require))
	for i, req := range data.parsed.Require {
		data.why[req.Mod.Path] = whyList[i]
	}
	return nil
}

func dependencyUpgrades(ctx context.Context, cfg *packages.Config, modFH, sumFH source.FileHandle, tmpMod bool, data *modData) error {
	if len(data.parsed.Require) == 0 {
		return nil
	}
	// Run "go list -mod readonly -u -m all" to be able to see which deps can be
	// upgraded without modifying mod file.
	args := []string{"-u", "-m", "all"}
	if !tmpMod || containsVendor(modFH.URI()) {
		// Use -mod=readonly if the module contains a vendor directory
		// (see golang/go#38711).
		args = append([]string{"-mod", "readonly"}, args...)
	}
	// If the -modfile flag is disabled, don't pass in a go.mod URI.
	if !tmpMod {
		modFH = nil
	}
	_, stdout, err := runGoCommand(ctx, cfg, modFH, sumFH, "list", args)
	if err != nil {
		return err
	}
	upgradesList := strings.Split(stdout.String(), "\n")
	if len(upgradesList) <= 1 {
		return nil
	}
	data.upgrades = make(map[string]string)
	for _, upgrade := range upgradesList[1:] {
		// Example: "github.com/x/tools v1.1.0 [v1.2.0]"
		info := strings.Split(upgrade, " ")
		if len(info) < 3 {
			continue
		}
		dep, version := info[0], info[2]
		latest := version[1:]                    // remove the "["
		latest = strings.TrimSuffix(latest, "]") // remove the "]"
		data.upgrades[dep] = latest
	}
	return nil
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
