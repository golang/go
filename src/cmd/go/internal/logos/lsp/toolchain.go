// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"internal/logosfile"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"sync"
)

type compiledPkg struct {
	Dir        string
	Standard   bool
	GoFiles    []string
	LogosFiles []compiledFile
	LogosHints json.RawMessage
	LogosDiags []json.RawMessage
}

type compiledFile struct {
	File         string
	LogosLowered string
	LogosEditMap json.RawMessage
}

type listRun struct {
	key   string
	pkgs  []compiledPkg
	diags map[string][]Diagnostic
	hints map[string]json.RawMessage
}

type toolchain struct {
	exe string

	mu         sync.Mutex
	last       *listRun
	generation uint64
}

func newToolchain(exe string) *toolchain {
	return &toolchain{exe: exe}
}

func (t *toolchain) DiskChanged() {
	t.mu.Lock()
	t.last = nil
	t.generation++
	t.mu.Unlock()
}

type overlay struct {
	json string
	back map[string]string
}

func newOverlay(buffers map[string][]byte) (*overlay, error) {
	root, err := os.MkdirTemp("", "logoslsp")
	if err != nil {
		return nil, err
	}
	ov := &overlay{json: filepath.Join(root, "overlay.json"), back: make(map[string]string, len(buffers))}
	replace := make(map[string]string, len(buffers))
	for i, path := range slices.Sorted(maps.Keys(buffers)) {
		dir := filepath.Join(root, strconv.Itoa(i))
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return nil, errors.Join(err, os.RemoveAll(root))
		}
		backing := filepath.Join(dir, filepath.Base(path))
		if err := os.WriteFile(backing, buffers[path], 0o600); err != nil {
			return nil, errors.Join(err, os.RemoveAll(root))
		}
		replace[path] = backing
		ov.back[backing] = path
	}
	data, err := json.Marshal(map[string]map[string]string{"Replace": replace})
	if err != nil {
		return nil, errors.Join(err, os.RemoveAll(root))
	}
	if err := os.WriteFile(ov.json, data, 0o600); err != nil {
		return nil, errors.Join(err, os.RemoveAll(root))
	}
	return ov, nil
}

func (ov *overlay) release() error {
	return os.RemoveAll(filepath.Dir(ov.json))
}

func (ov *overlay) source(file string) string {
	if path, ok := ov.back[file]; ok {
		return path
	}
	return file
}

func (t *toolchain) run(ctx context.Context, dir string, args ...string) (stdout, stderr []byte, err error) {
	cmd := exec.CommandContext(ctx, t.exe, args...)
	cmd.Args[0] = "logos"
	cmd.Dir = dir
	var so, se bytes.Buffer
	cmd.Stdout, cmd.Stderr = &so, &se
	err = cmd.Run()
	if _, ok := errors.AsType[*exec.ExitError](err); ok && ctx.Err() == nil {
		err = nil
	}
	return so.Bytes(), se.Bytes(), err
}

func runKey(dir string, buffers map[string][]byte) string {
	h := sha256.New()
	h.Write([]byte(filepath.Clean(dir)))
	h.Write([]byte{0})
	for _, path := range slices.Sorted(maps.Keys(buffers)) {
		h.Write([]byte(path))
		h.Write([]byte{0})
		h.Write(fmt.Appendf(nil, "%d:", len(buffers[path])))
		h.Write(buffers[path])
	}
	return string(h.Sum(nil))
}

func (t *toolchain) runOnce(ctx context.Context, dir string, buffers map[string][]byte) (*listRun, error) {
	key := runKey(dir, buffers)
	t.mu.Lock()
	last := t.last
	generation := t.generation
	t.mu.Unlock()
	if last != nil && last.key == key {
		return last, nil
	}
	ov, err := newOverlay(buffers)
	if err != nil {
		return nil, err
	}
	defer func() {
		if rerr := ov.release(); rerr != nil && err == nil {
			err = rerr
		}
	}()
	stdout, stderr, err := t.run(ctx, dir, "list", "-e", "-deps", "-export", "-compiled", "-buildvcs=false", "-json", "--json-diagnostics", "-overlay="+ov.json, ".")
	if err != nil {
		return nil, err
	}
	var pkgs []compiledPkg
	dec := json.NewDecoder(bytes.NewReader(stdout))
	for dec.More() {
		var p compiledPkg
		if err := dec.Decode(&p); err != nil {
			return nil, fmt.Errorf("decoding logos list -compiled output: %v", err)
		}
		pkgs = append(pkgs, p)
	}
	if len(pkgs) == 0 {
		return nil, fmt.Errorf("logos list printed no package: %s", bytes.TrimSpace(stderr[:min(len(stderr), 1024)]))
	}
	last = &listRun{
		key:   key,
		pkgs:  pkgs,
		diags: parseCompiledDiags(stderr, ov, pkgs),
		hints: make(map[string]json.RawMessage, len(pkgs)),
	}
	for _, p := range pkgs {
		if len(p.LogosHints) > 0 {
			last.hints[filepath.Clean(p.Dir)] = p.LogosHints
		}
	}
	t.mu.Lock()
	if t.generation == generation {
		t.last = last
	}
	t.mu.Unlock()
	return last, nil
}

func parseCompiledDiags(stderr []byte, ov *overlay, pkgs []compiledPkg) map[string][]Diagnostic {
	diags := make(map[string][]Diagnostic)
	add := func(line []byte) {
		d, ok := parseDiagnostic(line)
		if !ok {
			return
		}
		d.File = ov.source(d.File)
		dir := filepath.Dir(d.File)
		if d.RuleID == "" && hasDiagnostic(diags[dir], d) {
			return
		}
		diags[dir] = append(diags[dir], d)
	}
	for line := range bytes.SplitSeq(stderr, []byte("\n")) {
		add(line)
	}
	for _, p := range pkgs {
		for _, line := range p.LogosDiags {
			add(line)
		}
	}
	return diags
}

func hasDiagnostic(out []Diagnostic, d Diagnostic) bool {
	if d.RuleID != "" {
		return false
	}
	return slices.ContainsFunc(out, func(e Diagnostic) bool {
		return e.RuleID == "" && e.File == d.File && e.Span == d.Span && e.Severity == d.Severity && e.Message == d.Message
	})
}

func (t *toolchain) Lower(ctx context.Context, dir string, buffers map[string][]byte) (*Lowering, error) {
	last, err := t.runOnce(ctx, dir, buffers)
	if err != nil {
		return nil, err
	}
	low := &Lowering{}
	for _, p := range last.pkgs {
		if p.Standard {
			continue
		}
		low.Dirs = append(low.Dirs, filepath.Clean(p.Dir))
		names := make(map[string]compiledFile, len(p.LogosFiles))
		for _, f := range p.LogosFiles {
			names[f.File] = f
		}
		for _, name := range p.GoFiles {
			if !logosfile.Is(name) {
				continue
			}
			path := filepath.Join(p.Dir, name)
			f, ok := names[name]
			if !ok {
				low.Missing = append(low.Missing, path)
				continue
			}
			file, err := compiledLoweredFile(path, f, buffers)
			if err != nil {
				low.Missing = append(low.Missing, path)
				continue
			}
			low.Files = append(low.Files, file)
		}
	}
	return low, nil
}

func compiledLoweredFile(path string, f compiledFile, buffers map[string][]byte) (LoweredFile, error) {
	var file LoweredFile
	var err error
	file.Out = []byte(f.LogosLowered)
	if err := json.Unmarshal(f.LogosEditMap, &file.Map); err != nil {
		return LoweredFile{}, err
	}
	file.Src, err = sourceText(path, buffers)
	if err != nil {
		return LoweredFile{}, err
	}
	file.Path = path
	return file, nil
}

func sourceText(path string, buffers map[string][]byte) ([]byte, error) {
	if data, ok := buffers[path]; ok {
		return data, nil
	}
	return os.ReadFile(path)
}

func (t *toolchain) Diagnose(ctx context.Context, dir string, buffers map[string][]byte) ([]Diagnostic, error) {
	last, err := t.runOnce(ctx, dir, buffers)
	if err != nil {
		return nil, err
	}
	return last.diags[filepath.Clean(dir)], nil
}

func (t *toolchain) Hints(ctx context.Context, dir string, buffers map[string][]byte) (json.RawMessage, error) {
	last, err := t.runOnce(ctx, dir, buffers)
	if err != nil {
		return nil, err
	}
	return last.hints[filepath.Clean(dir)], nil
}
