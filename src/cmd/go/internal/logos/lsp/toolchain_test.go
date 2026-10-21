// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"encoding/json"
	"internal/testenv"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// realWorkspace writes a workspace of a loGo/s module (package a) and a Go
// module (packages b and plain, b importing a), the shape SPEC Sec 0.1
// allows, and returns the toolchain the tests drive: the go binary of the
// toolchain under test, run as logos, as logos lsp runs it.
func realWorkspace(t *testing.T) (root string, tc *toolchain) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	t.Setenv("GOFLAGS", "")
	t.Setenv("GOTOOLCHAIN", "local")
	t.Setenv("GOPROXY", "off")
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		"go.work":           "go 1.27\n\nuse (\n\t./lmod\n\t./gomod\n)\n",
		"lmod/go.mod":       "module example.com/lmod\n\ngo 1.27\n",
		"lmod/a/a.lgo":      aLgo,
		"lmod/app/main.lgo": "package main\n\nfunc main() {}\n",
		"gomod/go.mod":      "module example.com/gomod\n\ngo 1.27\n\nrequire example.com/lmod v0.0.0\n\nreplace example.com/lmod => ../lmod\n",
		"gomod/b/b.go":      "package b\n\nimport \"example.com/lmod/a\"\n\nvar _ = a.Read\n",
		"gomod/plain/p.go":  "package plain\n\nfunc F() int { return 1 }\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	tc, err = newToolchain(testenv.GoToolPath(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { tc.Close() })
	return root, tc
}

// TestToolchainLowersImportedLogosPackage runs the production compiler
// path over a Go package that imports a loGo/s package: the lowering holds
// the loGo/s file with an edit map that fits its texts, the ? statement's
// operand maps exactly, nothing lands in the workspace, and an unsaved
// buffer replaces the file on disk.
func TestToolchainLowersImportedLogosPackage(t *testing.T) {
	root, tc := realWorkspace(t)
	ctx := t.Context()
	aPath, bDir := filepath.Join(root, "lmod", "a", "a.lgo"), filepath.Join(root, "gomod", "b")

	low, err := tc.Lower(ctx, bDir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(low.Missing) != 0 || len(low.Files) != 1 || low.Files[0].Path != aPath {
		t.Fatalf("lowering = %+v, want the one file %s", low, aPath)
	}
	for _, dir := range []string{filepath.Dir(aPath), bDir} {
		if !slices.Contains(low.Dirs, dir) {
			t.Errorf("closure dirs %v lack %s", low.Dirs, dir)
		}
	}
	file := low.Files[0]
	if string(file.Src) != aLgo {
		t.Errorf("source read by the compiler = %q, want the file on disk", file.Src)
	}
	pm, err := newPosMap(file.Src, file.Out, file.Map)
	if err != nil {
		t.Fatalf("the compiler's edit map does not fit its texts: %v", err)
	}
	out := string(file.Out)
	if !strings.Contains(out, "os.ReadFile(path)") || strings.Contains(out, "?\n") {
		t.Errorf("lowered text does not hold the lowered statement:\n%s", out)
	}

	orig := shift(findPos(t, aLgo, "ReadFile(path)?", 0), len("ReadFile("))
	lowPos := pm.toLowered(orig, leading)
	if got := lineOf(out, lowPos.Line)[lowPos.Character:]; !strings.HasPrefix(got, "path)") {
		t.Errorf("operand %v maps to lowered %v, which starts %q", orig, lowPos, got)
	}
	if back := pm.toOriginal(lowPos, leading); back != orig {
		t.Errorf("operand round trip %v -> %v -> %v", orig, lowPos, back)
	}
	q := findPos(t, aLgo, "?", 0)
	end := findPos(t, aLgo, "return string(data)", 0)
	lowEnd := pm.toLowered(end, leading)
	if lines := lowEnd.Line - pm.toLowered(q, leading).Line; lines < 2 {
		t.Errorf("the ? statement expands to %d more lines in the lowered text, want at least 2", lines)
	}

	for _, stray := range []string{"lmod/a/a.go", "lmod/a/a.go.edits", "gomod/b/b.go.edits"} {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(stray))); err == nil {
			t.Errorf("lowering wrote %s into the workspace", stray)
		}
	}

	edited := "// unsaved\n" + aLgo
	low, err = tc.Lower(ctx, bDir, map[string][]byte{aPath: []byte(edited)})
	if err != nil {
		t.Fatal(err)
	}
	if len(low.Files) != 1 || string(low.Files[0].Src) != edited || !strings.Contains(string(low.Files[0].Out), "// unsaved") {
		t.Errorf("an unsaved buffer was not lowered: %+v", low)
	}
	onDisk, _ := os.ReadFile(aPath)
	if string(onDisk) != aLgo {
		t.Errorf("the file on disk changed to %q", onDisk)
	}
}

// TestToolchainWritesNoBinaryForMainPackage checks that lowering a main
// package leaves the workspace as it was: a build of one main package
// writes its executable into the current directory unless told otherwise,
// and the current directory here is the user's package directory.
func TestToolchainWritesNoBinaryForMainPackage(t *testing.T) {
	root, tc := realWorkspace(t)
	appDir := filepath.Join(root, "lmod", "app")
	before := listDir(t, appDir)
	low, err := tc.Lower(t.Context(), appDir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(appDir, "main.lgo"); len(low.Files) != 1 || low.Files[0].Path != want || len(low.Missing) != 0 {
		t.Fatalf("lowering = %+v, want the one file %s", low, want)
	}
	if after := listDir(t, appDir); !slices.Equal(before, after) {
		t.Errorf("the package directory held %v and holds %v after lowering", before, after)
	}
	bin, err := os.ReadDir(filepath.Join(tc.stage, "bin"))
	if err == nil && len(bin) != 0 {
		t.Errorf("the scratch executable directory keeps %d entries", len(bin))
	}
}

// listDir returns the sorted names in dir.
func listDir(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	slices.Sort(names)
	return names
}

// TestToolchainReportsBrokenSourceAsMissing checks that a package that
// does not compile yields no lowered file for its sources, and that
// Diagnose names the error under the source's own path, not under the
// scratch copy of an unsaved buffer.
func TestToolchainReportsBrokenSourceAsMissing(t *testing.T) {
	root, tc := realWorkspace(t)
	ctx := t.Context()
	aDir := filepath.Join(root, "lmod", "a")
	aPath := filepath.Join(aDir, "a.lgo")
	broken := strings.Replace(aLgo, "return string(data), nil", "x := 1\n\treturn string(data), nil", 1)

	low, err := tc.Lower(ctx, aDir, map[string][]byte{aPath: []byte(broken)})
	if err != nil {
		t.Fatal(err)
	}
	if len(low.Files) != 0 || !slices.Equal(low.Missing, []string{aPath}) {
		t.Errorf("lowering of a broken package = %+v, want %s missing", low, aPath)
	}
	diags, err := tc.Diagnose(ctx, aDir, map[string][]byte{aPath: []byte(broken)})
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, d := range diags {
		if d.File == aPath && strings.Contains(d.Message, "declared and not used") && d.Span.StartLine == 7 {
			found = true
		}
	}
	if !found {
		t.Errorf("diagnostics = %+v, want declared and not used at line 7 of %s", diags, aPath)
	}
	diags, err = tc.Diagnose(ctx, aDir, nil)
	if err != nil || len(diags) != 0 {
		t.Errorf("diagnostics of the good package = %+v, %v, want none", diags, err)
	}
}

// TestToolchainDiagnosesGoPackages checks that the same path serves a Go
// package: logos lint reports its errors as JSON diagnostics of the .go
// file.
func TestToolchainDiagnosesGoPackages(t *testing.T) {
	root, tc := realWorkspace(t)
	pDir := filepath.Join(root, "gomod", "plain")
	pPath := filepath.Join(pDir, "p.go")
	diags, err := tc.Diagnose(t.Context(), pDir, map[string][]byte{
		pPath: []byte("package plain\n\nfunc F() int { var y int; return 1 }\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(diags) != 1 || diags[0].File != pPath || !strings.Contains(diags[0].Message, "declared and not used") {
		t.Errorf("diagnostics = %+v, want one declared and not used in %s", diags, pPath)
	}
}

// TestToolchainHints runs the hints runner over a loGo/s package: the
// compiler writes its hints report and the runner returns it as JSON.
func TestToolchainHints(t *testing.T) {
	root, tc := realWorkspace(t)
	hints, err := tc.Hints(t.Context(), filepath.Join(root, "lmod", "a"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(hints) {
		t.Errorf("hints are not JSON: %s", hints)
	}
}

// TestImportConfigAndHintSources checks the two pure steps of the hints
// runner: the importcfg holds the dependencies' export data but not the
// target's own, and unsaved buffers are written under their base names.
func TestImportConfigAndHintSources(t *testing.T) {
	target := pkgInfo{importPath: "example.com/m/a", dir: "/w/a", files: []string{"a.lgo", "b.lgo"}, export: "/cache/a.a"}
	pkgs := []pkgInfo{
		{importPath: "fmt", export: "/cache/fmt.a", standard: true},
		{importPath: "broken"},
		target,
	}
	if got, want := string(importConfig(pkgs, target)), "packagefile fmt=/cache/fmt.a\n"; got != want {
		t.Errorf("importConfig = %q, want %q", got, want)
	}
	work := t.TempDir()
	files, err := hintSources(work, target, map[string][]byte{"/w/a/b.lgo": []byte("unsaved")})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"/w/a/a.lgo", filepath.Join(work, "src", "b.lgo")}; !slices.Equal(files, want) {
		t.Errorf("hintSources = %v, want %v", files, want)
	}
	if data, _ := os.ReadFile(files[1]); string(data) != "unsaved" {
		t.Errorf("the buffer copy holds %q", data)
	}
}

// TestHintsArchLevelsFollowsModuleTargetCPU checks that the hints compile
// environment follows the module's target_cpu and that an explicit user
// setting wins over the file: a baseline module selects the baseline
// levels, a module without logos.toml selects the deterministic levels,
// the file's levels reach the compile environment, and GOAMD64/GOARM64
// in the environment override the file.
func TestHintsArchLevelsFollowsModuleTargetCPU(t *testing.T) {
	root := t.TempDir()
	writeFile := func(dir, name, content string) {
		t.Helper()
		if err := os.MkdirAll(dir, 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	baseline := filepath.Join(root, "baseline")
	writeFile(baseline, "go.mod", "module example.com/base\n\ngo 1.27\n")
	writeFile(baseline, "logos.toml", "[build]\ntarget_cpu = \"baseline\"\n")
	if amd64, arm64 := hintsArchLevels(baseline); amd64 != "v1" || arm64 != "v8.0" {
		t.Errorf("baseline levels = (%q, %q), want (v1, v8.0)", amd64, arm64)
	}
	plain := filepath.Join(root, "plain")
	writeFile(plain, "go.mod", "module example.com/plain\n\ngo 1.27\n")
	if amd64, arm64 := hintsArchLevels(plain); amd64 != "v3" || arm64 != "v8.2" {
		t.Errorf("default levels = (%q, %q), want (v3, v8.2)", amd64, arm64)
	}
	// Without explicit settings the file's levels reach the compile
	// environment: empty process values count as unset here, as in
	// hintsEnv, so the file path is exercised deterministically.
	t.Setenv("GOAMD64", "")
	t.Setenv("GOARM64", "")
	if env := hintsEnv(baseline); !slices.Contains(env, "GOAMD64=v1") || !slices.Contains(env, "GOARM64=v8.0") {
		t.Errorf("file levels missing from hints env: %q", env)
	}
	// An explicit user setting wins over the file: the file's levels
	// must not reach the compile environment.
	t.Setenv("GOAMD64", "v4")
	t.Setenv("GOARM64", "v8.1")
	if env := hintsEnv(baseline); !slices.Contains(env, "GOAMD64=v4") || !slices.Contains(env, "GOARM64=v8.1") ||
		slices.Contains(env, "GOAMD64=v1") || slices.Contains(env, "GOARM64=v8.0") {
		t.Errorf("explicit environment lost in hints env: %q", env)
	}
}

// TestExecutableDirFlag checks when the executable directory is asked for:
// only for a build of exactly one main package, the one build that writes
// an executable into the current directory.
func TestExecutableDirFlag(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "bin")
	lib, app := pkgInfo{importPath: "m/lib", name: "lib"}, pkgInfo{importPath: "m/app", name: "main"}
	cases := []struct {
		name    string
		targets []pkgInfo
		want    bool
	}{
		{"one main package", []pkgInfo{app}, true},
		{"one library", []pkgInfo{lib}, false},
		{"a library and a main package", []pkgInfo{lib, app}, false},
		{"two main packages", []pkgInfo{app, app}, false},
	}
	for _, c := range cases {
		flag, err := executableDirFlag(c.targets, bin)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got := len(flag) > 0; got != c.want {
			t.Errorf("%s: flag = %v, want one: %v", c.name, flag, c.want)
		}
		if c.want && (len(flag) != 2 || flag[0] != "-o" || flag[1] != bin+string(filepath.Separator)) {
			t.Errorf("%s: flag = %v, want -o %s/", c.name, flag, bin)
		}
	}
}

// TestParseClosure checks the list output parser, including a package with
// no files and a malformed line.
func TestParseClosure(t *testing.T) {
	out := "fmt\t/g/fmt\tfmt\ttrue\t/c/fmt.a\tprint.go\tformat.go\nexample.com/m/a\t/w/a\ta\tfalse\t\ta.lgo\nexample.com/m/empty\t/w/e\tmain\tfalse\t\n"
	pkgs, err := parseClosure([]byte(out))
	if err != nil {
		t.Fatal(err)
	}
	if len(pkgs) != 3 || !pkgs[0].standard || pkgs[0].export != "/c/fmt.a" || !slices.Equal(pkgs[0].files, []string{"print.go", "format.go"}) {
		t.Errorf("packages = %+v", pkgs)
	}
	if pkgs[1].standard || pkgs[1].name != "a" || pkgs[1].export != "" || !slices.Equal(pkgs[1].logosFiles(), []string{"a.lgo"}) || len(pkgs[2].files) != 0 || pkgs[2].name != "main" {
		t.Errorf("packages = %+v", pkgs)
	}
	if _, err := parseClosure([]byte("only\ttwo\tthree\tfour\n")); err == nil {
		t.Error("parseClosure accepted a short line")
	}
}
