// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/config"
	"cmd/internal/quoted"
)

// listTemplate prints one tab-separated line for each package of the
// dependency closure: import path, directory, package name, whether it
// belongs to the standard library, the export data file (empty unless the
// list was asked to build exports), then the file names of the package,
// which include its loGo/s sources.
const listTemplate = `{{.ImportPath}}{{"\t"}}{{.Dir}}{{"\t"}}{{.Name}}{{"\t"}}{{.Standard}}{{"\t"}}{{.Export}}{{range .GoFiles}}{{"\t"}}{{.}}{{end}}{{"\n"}}`

// toolchain is the production Compiler. It runs the logos command, the
// executable of the process itself, as a subprocess for every request, so
// a fatal error of the go command never takes the proxy down. Unsaved
// buffers reach the subprocess through its -overlay flag; lowered text
// comes from the -logos.eject compiler flag, which the build passes to the
// loGo/s packages of the closure only, and which makes those packages skip
// the build cache.
type toolchain struct {
	exe   string
	stage string

	mu  sync.Mutex
	gen int
}

// newToolchain returns a toolchain that runs exe and keeps its scratch
// files in a fresh directory under the system temporary directory.
func newToolchain(exe string) (*toolchain, error) {
	stage, err := os.MkdirTemp("", "logoslsp")
	if err != nil {
		return nil, err
	}
	return &toolchain{exe: exe, stage: stage}, nil
}

// Close removes the scratch directory.
func (t *toolchain) Close() error { return os.RemoveAll(t.stage) }

// pkgInfo is one package of a dependency closure.
type pkgInfo struct {
	importPath string
	dir        string
	name       string
	standard   bool
	export     string
	files      []string
}

// logosFiles returns the loGo/s source files of p.
func (p pkgInfo) logosFiles() []string {
	return slices.DeleteFunc(slices.Clone(p.files), func(name string) bool { return !cfg.IsLogosFile(name) })
}

// overlay is the -overlay file of one run, with the backing copies of the
// buffers it replaces.
type overlay struct {
	root string
	json string
	back map[string]string
}

// newOverlay writes the buffers to backing files and the overlay JSON
// that maps each source path to its backing file. A backing file keeps the
// base name of its source, so the compiler still sees a loGo/s file and
// names its output after the source; each lives in a directory of its own
// so equal base names cannot collide.
func (t *toolchain) newOverlay(buffers map[string][]byte) (*overlay, error) {
	t.mu.Lock()
	t.gen++
	root := filepath.Join(t.stage, "buf", strconv.Itoa(t.gen))
	t.mu.Unlock()
	ov := &overlay{root: root, json: filepath.Join(root, "overlay.json"), back: make(map[string]string, len(buffers))}
	replace := make(map[string]string, len(buffers))
	if err := os.MkdirAll(root, 0o777); err != nil {
		return nil, err
	}
	for i, path := range slices.Sorted(maps.Keys(buffers)) {
		dir := filepath.Join(root, strconv.Itoa(i))
		if err := os.MkdirAll(dir, 0o777); err != nil {
			return nil, err
		}
		backing := filepath.Join(dir, filepath.Base(path))
		if err := os.WriteFile(backing, buffers[path], 0o666); err != nil {
			return nil, err
		}
		replace[path] = backing
		ov.back[backing] = path
	}
	data, err := json.Marshal(map[string]map[string]string{"Replace": replace})
	if err != nil {
		return nil, err
	}
	return ov, os.WriteFile(ov.json, data, 0o666)
}

// release removes the overlay files.
func (ov *overlay) release() { os.RemoveAll(ov.root) }

// source returns the source path a compiler path refers to: the path of
// the buffer when file is one of the backing files, else file itself.
func (ov *overlay) source(file string) string {
	if path, ok := ov.back[file]; ok {
		return path
	}
	return file
}

// run executes the logos command with args in dir and returns what it
// wrote to stdout and stderr. The command runs as "logos" whatever the
// executable is named, so it keeps logos mode. A command that ran and
// failed is not an error here: its diagnostics are the answer.
func (t *toolchain) run(ctx context.Context, dir string, args ...string) (stdout, stderr []byte, err error) {
	return t.runEnv(ctx, dir, nil, args...)
}

// runEnv is run with cmd.Env set to env. A nil env inherits the process
// environment.
func (t *toolchain) runEnv(ctx context.Context, dir string, env []string, args ...string) (stdout, stderr []byte, err error) {
	cmd := exec.CommandContext(ctx, t.exe, args...)
	cmd.Args[0] = "logos"
	cmd.Dir = dir
	cmd.Env = env
	var so, se bytes.Buffer
	cmd.Stdout, cmd.Stderr = &so, &se
	err = cmd.Run()
	if _, ok := errors.AsType[*exec.ExitError](err); ok && ctx.Err() == nil {
		err = nil
	}
	return so.Bytes(), se.Bytes(), err
}

// closure lists the packages the package in dir depends on, itself
// included, with the flags in extra added to the list command.
func (t *toolchain) closure(ctx context.Context, dir string, ov *overlay, extra ...string) ([]pkgInfo, error) {
	args := append([]string{"list", "-e", "-deps", "-overlay=" + ov.json, "-f", listTemplate}, extra...)
	out, stderr, err := t.run(ctx, dir, append(args, ".")...)
	if err != nil {
		return nil, err
	}
	pkgs, err := parseClosure(out)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", err, strings.TrimSpace(string(stderr)))
	}
	return pkgs, nil
}

// parseClosure parses the output of listTemplate.
func parseClosure(out []byte) ([]pkgInfo, error) {
	var pkgs []pkgInfo
	sc := bufio.NewScanner(bytes.NewReader(out))
	sc.Buffer(nil, maxMessage)
	for sc.Scan() {
		fields := strings.Split(sc.Text(), "\t")
		if len(fields) < 5 {
			return nil, fmt.Errorf("unexpected list output %q", sc.Text())
		}
		pkgs = append(pkgs, pkgInfo{
			importPath: fields[0],
			dir:        filepath.Clean(fields[1]),
			name:       fields[2],
			standard:   fields[3] == "true",
			export:     fields[4],
			files:      fields[5:],
		})
	}
	return pkgs, sc.Err()
}

// ejectDir returns the directory the compiler writes the lowered files of
// the package importPath to. It is the same for every run, so the scratch
// directory does not grow, and the build cache cannot hold stale output
// because eject packages never use it.
func (t *toolchain) ejectDir(importPath string) string {
	sum := sha256.Sum256([]byte(importPath))
	return filepath.Join(t.stage, "pkg", hex.EncodeToString(sum[:8]))
}

// gcflag returns the -gcflags argument that gives flags to the package
// importPath only.
func gcflag(importPath string, flags ...string) (string, error) {
	joined, err := quoted.Join(flags)
	if err != nil {
		return "", err
	}
	return "-gcflags=" + importPath + "=" + joined, nil
}

// Lower implements Compiler.
func (t *toolchain) Lower(ctx context.Context, dir string, buffers map[string][]byte) (*Lowering, error) {
	ov, err := t.newOverlay(buffers)
	if err != nil {
		return nil, err
	}
	defer ov.release()
	pkgs, err := t.closure(ctx, dir, ov)
	if err != nil {
		return nil, err
	}
	low := &Lowering{}
	var targets []pkgInfo
	for _, p := range pkgs {
		if p.standard {
			continue
		}
		low.Dirs = append(low.Dirs, p.dir)
		if len(p.logosFiles()) > 0 {
			targets = append(targets, p)
		}
	}
	if err := t.eject(ctx, dir, ov, targets); err != nil {
		return nil, err
	}
	for _, p := range targets {
		t.collect(low, p, buffers)
	}
	return low, nil
}

// eject builds the packages of targets with the eject flag, so the
// compiler writes their lowered files. A package that fails to compile
// writes nothing; the caller finds out by its missing files.
func (t *toolchain) eject(ctx context.Context, dir string, ov *overlay, targets []pkgInfo) error {
	if len(targets) == 0 {
		return nil
	}
	bin := filepath.Join(t.stage, "bin")
	defer os.RemoveAll(bin)
	output, err := executableDirFlag(targets, bin)
	if err != nil {
		return err
	}
	args := append([]string{"build", "-overlay=" + ov.json}, output...)
	for _, p := range targets {
		out := t.ejectDir(p.importPath)
		if err := os.RemoveAll(out); err != nil {
			return err
		}
		flag, err := gcflag(p.importPath, "-logos.eject="+out)
		if err != nil {
			return err
		}
		args = append(args, flag)
	}
	for _, p := range targets {
		args = append(args, p.importPath)
	}
	_, _, err = t.run(ctx, dir, args...)
	return err
}

// executableDirFlag returns the -o flag that sends executables to bin when
// the build would write one. A build of a single main package writes its
// executable into the current directory, the user's package directory;
// every other build writes none. The directory form of -o is an error when
// no main package is built, so the flag is given for that case alone.
func executableDirFlag(targets []pkgInfo, bin string) ([]string, error) {
	if len(targets) != 1 || targets[0].name != "main" {
		return nil, nil
	}
	if err := os.MkdirAll(bin, 0o777); err != nil {
		return nil, err
	}
	return []string{"-o", bin + string(filepath.Separator)}, nil
}

// collect reads the lowered files of package p into low. A file is
// missing from the result when its output or edit map is absent or
// unreadable.
func (t *toolchain) collect(low *Lowering, p pkgInfo, buffers map[string][]byte) {
	out := t.ejectDir(p.importPath)
	for _, name := range p.logosFiles() {
		path := filepath.Join(p.dir, name)
		file, err := readLowered(out, path, buffers)
		if err != nil {
			low.Missing = append(low.Missing, path)
			continue
		}
		low.Files = append(low.Files, file)
	}
}

// readLowered reads the lowered text and edit map the compiler wrote under
// out for the source at path, and the source text the compiler read.
func readLowered(out, path string, buffers map[string][]byte) (LoweredFile, error) {
	ejected := filepath.Join(out, filepath.Base(virtualPath(path)))
	text, err := os.ReadFile(ejected)
	if err != nil {
		return LoweredFile{}, err
	}
	raw, err := os.ReadFile(ejected + ".edits")
	if err != nil {
		return LoweredFile{}, err
	}
	file := LoweredFile{Path: path, Out: text}
	if err := json.Unmarshal(raw, &file.Map); err != nil {
		return LoweredFile{}, err
	}
	file.Src, err = sourceText(path, buffers)
	return file, err
}

// sourceText returns the text of the source at path: the buffer when the
// document is open, else the file on disk.
func sourceText(path string, buffers map[string][]byte) ([]byte, error) {
	if data, ok := buffers[path]; ok {
		return data, nil
	}
	return os.ReadFile(path)
}

// Diagnose implements Compiler. It runs logos lint, which type-checks the
// package and runs the compiler lints without building a binary.
func (t *toolchain) Diagnose(ctx context.Context, dir string, buffers map[string][]byte) ([]Diagnostic, error) {
	ov, err := t.newOverlay(buffers)
	if err != nil {
		return nil, err
	}
	defer ov.release()
	_, stderr, err := t.run(ctx, dir, "lint", "--json-diagnostics", "-overlay="+ov.json, ".")
	if err != nil {
		return nil, err
	}
	var diags []Diagnostic
	for line := range bytes.SplitSeq(stderr, []byte("\n")) {
		if d, ok := parseDiagnostic(line); ok {
			d.File = ov.source(d.File)
			diags = append(diags, d)
		}
	}
	return diags, nil
}

// Hints returns the hints JSON of the package in dir, the file the
// compiler writes under -logos.mode=hints -logos.hintsout=FILE. The mode
// stops before code generation and writes no object file, and the build
// cache would answer a hit without running the compiler, so the compiler
// runs directly: the export data of the package's imports comes from
// logos list -export, and the sources are the package's own files, with
// the text of unsaved buffers in place of the files on disk.
func (t *toolchain) Hints(ctx context.Context, dir string, buffers map[string][]byte) (json.RawMessage, error) {
	ov, err := t.newOverlay(buffers)
	if err != nil {
		return nil, err
	}
	defer ov.release()
	pkgs, err := t.closure(ctx, dir, ov, "-export")
	if err != nil {
		return nil, err
	}
	i := slices.IndexFunc(pkgs, func(p pkgInfo) bool { return p.dir == filepath.Clean(dir) && !p.standard })
	if i < 0 {
		return nil, fmt.Errorf("no package in %s", dir)
	}
	work, err := os.MkdirTemp(t.stage, "hints")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(work)
	return t.compileHints(ctx, work, pkgs[i], pkgs, buffers)
}

// compileHints writes the importcfg and sources of target under work and
// runs the compiler in hints mode.
func (t *toolchain) compileHints(ctx context.Context, work string, target pkgInfo, pkgs []pkgInfo, buffers map[string][]byte) (json.RawMessage, error) {
	importcfg := filepath.Join(work, "importcfg")
	if err := os.WriteFile(importcfg, importConfig(pkgs, target), 0o666); err != nil {
		return nil, err
	}
	files, err := hintSources(work, target, buffers)
	if err != nil {
		return nil, err
	}
	out := filepath.Join(work, "hints.json")
	args := []string{"tool", "compile", "-p", target.importPath, "-importcfg", importcfg,
		"-logos.mode=hints", "-logos.hintsout=" + out, "-o", filepath.Join(work, "x.a")}
	stdout, stderr, err := t.runEnv(ctx, target.dir, hintsEnv(target.dir), append(args, files...)...)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(out)
	if err != nil {
		detail := strings.TrimSpace(string(stderr) + "\n" + string(stdout))
		return nil, fmt.Errorf("no hints for %s: %s", target.importPath, detail)
	}
	return json.RawMessage(data), nil
}

// hintsEnv returns the environment for the direct hints compile: the
// process environment plus the GOAMD64 and GOARM64 levels logos list and
// build compiled the importcfg exports with. A bare go tool compile
// defaults to v1, while a logos build defaults to the deterministic
// target_cpu (v3); the mismatch makes every import unreadable, so the
// compile prints to stdout, writes no report, and Hints fails with an
// empty message. An explicit user setting wins over the file.
func hintsEnv(dir string) []string {
	amd64, arm64 := hintsArchLevels(dir)
	// An explicit user setting wins over the file, whether it is
	// exported or lives only in the GOENV file: the bare compiler
	// reads neither GOENV nor logos.toml, so a GOENV-only value must
	// be materialized here or the compile still defaults to v1.
	if v := cfg.Getenv("GOAMD64"); v != "" {
		amd64 = v
	}
	if v := cfg.Getenv("GOARM64"); v != "" {
		arm64 = v
	}
	env := os.Environ()
	if os.Getenv("GOAMD64") == "" {
		env = append(env, "GOAMD64="+amd64)
	}
	if os.Getenv("GOARM64") == "" {
		env = append(env, "GOARM64="+arm64)
	}
	return env
}

// hintsArchLevels returns the GOAMD64 and GOARM64 levels for the module
// holding dir. The selection is config.ArchLevels, the same owner the
// build path calls; a load failure falls back to its deterministic
// default, matching the build's treatment of a broken file.
func hintsArchLevels(dir string) (amd64, arm64 string) {
	file, err := config.Load(moduleRoot(dir))
	if err != nil || file == nil {
		return config.ArchLevels("")
	}
	return config.ArchLevels(file.Build.TargetCPU)
}

// moduleRoot returns the directory holding the go.mod enclosing dir, or
// "" when there is none, so config.Load yields its defaults. It walks
// target.dir itself, not the process working directory, which is the
// test binary's directory under logos lsp.
func moduleRoot(dir string) string {
	for d := dir; ; {
		if _, err := os.Stat(filepath.Join(d, "go.mod")); err == nil {
			return d
		}
		parent := filepath.Dir(d)
		if parent == d {
			return ""
		}
		d = parent
	}
}

// importConfig returns the importcfg text that maps every dependency of
// target that has export data to its file.
func importConfig(pkgs []pkgInfo, target pkgInfo) []byte {
	var buf bytes.Buffer
	for _, p := range pkgs {
		if p.export != "" && p.importPath != target.importPath {
			fmt.Fprintf(&buf, "packagefile %s=%s\n", p.importPath, p.export)
		}
	}
	return buf.Bytes()
}

// hintSources returns the paths of the source files of target for the
// compiler. A file with an unsaved buffer is written under work with its
// own base name, since the hints name their file by base name.
func hintSources(work string, target pkgInfo, buffers map[string][]byte) ([]string, error) {
	src := filepath.Join(work, "src")
	if err := os.MkdirAll(src, 0o777); err != nil {
		return nil, err
	}
	files := make([]string, 0, len(target.files))
	for _, name := range target.files {
		path := filepath.Join(target.dir, name)
		if data, ok := buffers[path]; ok {
			path = filepath.Join(src, name)
			if err := os.WriteFile(path, data, 0o666); err != nil {
				return nil, err
			}
		}
		files = append(files, path)
	}
	return files, nil
}
