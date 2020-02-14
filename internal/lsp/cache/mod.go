// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

const (
	ModTidyError = "go mod tidy"
	SyntaxError  = "syntax"
)

type parseModKey struct {
	view  string
	gomod string
	cfg   string
}

type parseModHandle struct {
	handle *memoize.Handle
	file   source.FileHandle
	cfg    *packages.Config
}

type modTidyKey struct {
	view    string
	imports string
	gomod   string
	cfg     string
}

type modTidyHandle struct {
	handle *memoize.Handle
	file   source.FileHandle
	cfg    *packages.Config
}

type modTidyData struct {
	memoize.NoCopy

	// origfh is the file handle for the original go.mod file.
	origfh source.FileHandle

	// origParsedFile contains the parsed contents that are used to diff with
	// the ideal contents.
	origParsedFile *modfile.File

	// origMapper is the column mapper for the original go.mod file.
	origMapper *protocol.ColumnMapper

	// idealParsedFile contains the parsed contents for the go.mod file
	// after it has been "tidied".
	idealParsedFile *modfile.File

	// unusedDeps is the map containing the dependencies that are left after
	// removing the ones that are identical in the original and ideal go.mods.
	unusedDeps map[string]*modfile.Require

	// missingDeps is the map containing the dependencies that are left after
	// removing the ones that are identical in the original and ideal go.mods.
	missingDeps map[string]*modfile.Require

	// upgrades is a map of path->version that contains any upgrades for the go.mod.
	upgrades map[string]string

	// parseErrors are the errors that arise when we diff between a user's go.mod
	// and the "tidied" go.mod.
	parseErrors []source.Error

	// err is any error that occurs while we are calculating the parseErrors.
	err error
}

func (pmh *parseModHandle) String() string {
	return pmh.File().Identity().URI.Filename()
}

func (pmh *parseModHandle) File() source.FileHandle {
	return pmh.file
}

func (pmh *parseModHandle) Upgrades(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, map[string]string, error) {
	v := pmh.handle.Get(ctx)
	if v == nil {
		return nil, nil, nil, errors.Errorf("no parsed file for %s", pmh.File().Identity().URI)
	}
	data := v.(*modTidyData)
	return data.origParsedFile, data.origMapper, data.upgrades, data.err
}

func (s *snapshot) ParseModHandle(ctx context.Context) (source.ParseModHandle, error) {
	cfg := s.Config(ctx)
	folder := s.View().Folder().Filename()

	realURI, tempURI := s.view.ModFiles()
	fh, err := s.GetFile(realURI)
	if err != nil {
		return nil, err
	}

	key := parseModKey{
		view:  folder,
		gomod: fh.Identity().String(),
		cfg:   hashConfig(cfg),
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		ctx, done := trace.StartSpan(ctx, "cache.ParseModHandle", telemetry.File.Of(realURI))
		defer done()

		data := &modTidyData{}
		contents, _, err := fh.Read(ctx)
		if err != nil {
			data.err = err
			return data
		}
		parsedFile, err := modfile.Parse(realURI.Filename(), contents, nil)
		if err != nil {
			data.err = err
			return data
		}
		// If we have a tempModfile, copy the real go.mod file content into the temp go.mod file.
		if tempURI != "" {
			if err := ioutil.WriteFile(tempURI.Filename(), contents, os.ModePerm); err != nil {
				data.err = err
				return data
			}
		}
		data = &modTidyData{
			origfh:         fh,
			origParsedFile: parsedFile,
			origMapper: &protocol.ColumnMapper{
				URI:       realURI,
				Converter: span.NewContentConverter(realURI.Filename(), contents),
				Content:   contents,
			},
		}
		data.upgrades, data.err = dependencyUpgrades(ctx, cfg, folder, data)
		return data
	})
	return &parseModHandle{
		handle: h,
		file:   fh,
		cfg:    cfg,
	}, nil
}

func dependencyUpgrades(ctx context.Context, cfg *packages.Config, folder string, modData *modTidyData) (map[string]string, error) {
	if len(modData.origParsedFile.Require) == 0 {
		return nil, nil
	}
	// Run "go list -u -m all" to be able to see which deps can be upgraded.
	args := []string{"list"}
	args = append(args, cfg.BuildFlags...)
	args = append(args, []string{"-u", "-m", "all"}...)
	stdout, err := source.InvokeGo(ctx, folder, cfg.Env, args...)
	if err != nil {
		return nil, err
	}
	upgradesList := strings.Split(stdout.String(), "\n")
	if len(upgradesList) <= 1 {
		return nil, nil
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
	return upgrades, nil
}

func (mth *modTidyHandle) String() string {
	return mth.File().Identity().URI.Filename()
}

func (mth *modTidyHandle) File() source.FileHandle {
	return mth.file
}

func (mth *modTidyHandle) Tidy(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, map[string]*modfile.Require, []source.Error, error) {
	v := mth.handle.Get(ctx)
	if v == nil {
		return nil, nil, nil, nil, errors.Errorf("no parsed file for %s", mth.File().Identity().URI)
	}
	data := v.(*modTidyData)
	return data.origParsedFile, data.origMapper, data.missingDeps, data.parseErrors, data.err
}

func (s *snapshot) ModTidyHandle(ctx context.Context, realfh source.FileHandle) (source.ModTidyHandle, error) {
	realURI, tempURI := s.view.ModFiles()
	cfg := s.Config(ctx)
	options := s.View().Options()
	folder := s.View().Folder().Filename()

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
	key := modTidyKey{
		view:    folder,
		imports: imports,
		gomod:   realfh.Identity().Identifier,
		cfg:     hashConfig(cfg),
	}
	h := s.view.session.cache.store.Bind(key, func(ctx context.Context) interface{} {
		data := &modTidyData{}

		// Check the case when the tempModfile flag is turned off.
		if realURI == "" || tempURI == "" {
			return data
		}

		ctx, done := trace.StartSpan(ctx, "cache.ModTidyHandle", telemetry.File.Of(realURI))
		defer done()

		realContents, _, err := realfh.Read(ctx)
		if err != nil {
			data.err = err
			return data
		}
		realMapper := &protocol.ColumnMapper{
			URI:       realURI,
			Converter: span.NewContentConverter(realURI.Filename(), realContents),
			Content:   realContents,
		}
		origParsedFile, err := modfile.Parse(realURI.Filename(), realContents, nil)
		if err != nil {
			if parseErr, err := extractModParseErrors(ctx, realURI, realMapper, err, realContents); err == nil {
				data.parseErrors = []source.Error{parseErr}
				return data
			}
			data.err = err
			return data
		}

		// Copy the real go.mod file content into the temp go.mod file.
		if err := ioutil.WriteFile(tempURI.Filename(), realContents, os.ModePerm); err != nil {
			data.err = err
			return data
		}

		// We want to run "go mod tidy" to be able to diff between the real and the temp files.
		args := append([]string{"mod", "tidy"}, cfg.BuildFlags...)
		if _, err := source.InvokeGo(ctx, folder, cfg.Env, args...); err != nil {
			// Ignore concurrency errors here.
			if !modConcurrencyError.MatchString(err.Error()) {
				data.err = err
				return data
			}
		}

		// Go directly to disk to get the temporary mod file, since it is always on disk.
		tempContents, err := ioutil.ReadFile(tempURI.Filename())
		if err != nil {
			data.err = err
			return data
		}
		idealParsedFile, err := modfile.Parse(tempURI.Filename(), tempContents, nil)
		if err != nil {
			// We do not need to worry about the temporary file's parse errors since it has been "tidied".
			data.err = err
			return data
		}

		data = &modTidyData{
			origfh:          realfh,
			origParsedFile:  origParsedFile,
			origMapper:      realMapper,
			idealParsedFile: idealParsedFile,
			unusedDeps:      make(map[string]*modfile.Require, len(origParsedFile.Require)),
			missingDeps:     make(map[string]*modfile.Require, len(idealParsedFile.Require)),
		}
		// Get the dependencies that are different between the original and ideal mod files.
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
		data.parseErrors, data.err = modRequireErrors(ctx, options, data)

		for _, req := range data.missingDeps {
			if data.unusedDeps[req.Mod.Path] != nil {
				delete(data.missingDeps, req.Mod.Path)
			}
		}
		return data
	})
	return &modTidyHandle{
		handle: h,
		file:   realfh,
		cfg:    cfg,
	}, nil
}

// extractModParseErrors processes the raw errors returned by modfile.Parse,
// extracting the filenames and line numbers that correspond to the errors.
func extractModParseErrors(ctx context.Context, uri span.URI, m *protocol.ColumnMapper, parseErr error, content []byte) (source.Error, error) {
	re := regexp.MustCompile(`.*:([\d]+): (.+)`)
	matches := re.FindStringSubmatch(strings.TrimSpace(parseErr.Error()))
	if len(matches) < 3 {
		log.Error(ctx, "could not parse golang/x/mod error message", parseErr)
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
func modRequireErrors(ctx context.Context, options source.Options, modData *modTidyData) ([]source.Error, error) {
	var errors []source.Error
	for dep, req := range modData.unusedDeps {
		if req.Syntax == nil {
			continue
		}
		// Handle dependencies that are incorrectly labeled indirect and vice versa.
		if modData.missingDeps[dep] != nil && req.Indirect != modData.missingDeps[dep].Indirect {
			directErr, err := modDirectnessErrors(ctx, options, modData, req)
			if err != nil {
				return nil, err
			}
			errors = append(errors, directErr)
		}
		// Handle unused dependencies.
		if modData.missingDeps[dep] == nil {
			rng, err := rangeFromPositions(modData.origfh.Identity().URI, modData.origMapper, req.Syntax.Start, req.Syntax.End)
			if err != nil {
				return nil, err
			}
			edits, err := dropDependencyEdits(ctx, options, modData, req)
			if err != nil {
				return nil, err
			}
			errors = append(errors, source.Error{
				Category: ModTidyError,
				Message:  fmt.Sprintf("%s is not used in this module.", dep),
				Range:    rng,
				URI:      modData.origfh.Identity().URI,
				SuggestedFixes: []source.SuggestedFix{{
					Title: fmt.Sprintf("Remove dependency: %s", dep),
					Edits: map[span.URI][]protocol.TextEdit{modData.origfh.Identity().URI: edits},
				}},
			})
		}
	}
	return errors, nil
}

// modDirectnessErrors extracts errors when a dependency is labeled indirect when it should be direct and vice versa.
func modDirectnessErrors(ctx context.Context, options source.Options, modData *modTidyData, req *modfile.Require) (source.Error, error) {
	rng, err := rangeFromPositions(modData.origfh.Identity().URI, modData.origMapper, req.Syntax.Start, req.Syntax.End)
	if err != nil {
		return source.Error{}, err
	}
	if req.Indirect {
		// If the dependency should be direct, just highlight the // indirect.
		if comments := req.Syntax.Comment(); comments != nil && len(comments.Suffix) > 0 {
			end := comments.Suffix[0].Start
			end.LineRune += len(comments.Suffix[0].Token)
			end.Byte += len([]byte(comments.Suffix[0].Token))
			rng, err = rangeFromPositions(modData.origfh.Identity().URI, modData.origMapper, comments.Suffix[0].Start, end)
			if err != nil {
				return source.Error{}, err
			}
		}
		edits, err := changeDirectnessEdits(ctx, options, modData, req, false)
		if err != nil {
			return source.Error{}, err
		}
		return source.Error{
			Category: ModTidyError,
			Message:  fmt.Sprintf("%s should be a direct dependency.", req.Mod.Path),
			Range:    rng,
			URI:      modData.origfh.Identity().URI,
			SuggestedFixes: []source.SuggestedFix{{
				Title: fmt.Sprintf("Make %s direct", req.Mod.Path),
				Edits: map[span.URI][]protocol.TextEdit{modData.origfh.Identity().URI: edits},
			}},
		}, nil
	}
	// If the dependency should be indirect, add the // indirect.
	edits, err := changeDirectnessEdits(ctx, options, modData, req, true)
	if err != nil {
		return source.Error{}, err
	}
	return source.Error{
		Category: ModTidyError,
		Message:  fmt.Sprintf("%s should be an indirect dependency.", req.Mod.Path),
		Range:    rng,
		URI:      modData.origfh.Identity().URI,
		SuggestedFixes: []source.SuggestedFix{{
			Title: fmt.Sprintf("Make %s indirect", req.Mod.Path),
			Edits: map[span.URI][]protocol.TextEdit{modData.origfh.Identity().URI: edits},
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
func dropDependencyEdits(ctx context.Context, options source.Options, modData *modTidyData, req *modfile.Require) ([]protocol.TextEdit, error) {
	if err := modData.origParsedFile.DropRequire(req.Mod.Path); err != nil {
		return nil, err
	}
	modData.origParsedFile.Cleanup()
	newContents, err := modData.origParsedFile.Format()
	if err != nil {
		return nil, err
	}
	// Reset the *modfile.File back to before we dropped the dependency.
	modData.origParsedFile.AddNewRequire(req.Mod.Path, req.Mod.Version, req.Indirect)
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(modData.origfh.Identity().URI, string(modData.origMapper.Content), string(newContents))
	edits, err := source.ToProtocolEdits(modData.origMapper, diff)
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
func changeDirectnessEdits(ctx context.Context, options source.Options, modData *modTidyData, req *modfile.Require, indirect bool) ([]protocol.TextEdit, error) {
	var newReq []*modfile.Require
	prevIndirect := false
	// Change the directness in the matching require statement.
	for _, r := range modData.origParsedFile.Require {
		if req.Mod.Path == r.Mod.Path {
			prevIndirect = req.Indirect
			req.Indirect = indirect
		}
		newReq = append(newReq, r)
	}
	modData.origParsedFile.SetRequire(newReq)
	modData.origParsedFile.Cleanup()
	newContents, err := modData.origParsedFile.Format()
	if err != nil {
		return nil, err
	}
	// Change the dependency back to the way it was before we got the newContents.
	for _, r := range modData.origParsedFile.Require {
		if req.Mod.Path == r.Mod.Path {
			req.Indirect = prevIndirect
		}
		newReq = append(newReq, r)
	}
	modData.origParsedFile.SetRequire(newReq)
	// Calculate the edits to be made due to the change.
	diff := options.ComputeEdits(modData.origfh.Identity().URI, string(modData.origMapper.Content), string(newContents))
	edits, err := source.ToProtocolEdits(modData.origMapper, diff)
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
