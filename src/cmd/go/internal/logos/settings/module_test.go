// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package settings

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"cmd/go/internal/logos/config"
)

func scanTree(t *testing.T, files map[string]string, dirs []string) string {
	t.Helper()
	dir := t.TempDir()
	for _, d := range dirs {
		if err := os.MkdirAll(filepath.Join(dir, filepath.FromSlash(d)), 0o777); err != nil {
			t.Fatal(err)
		}
	}
	for name, content := range files {
		path := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestModuleScanSkips(t *testing.T) {
	tests := []struct {
		name  string
		files map[string]string
		dirs  []string
		want  bool
	}{
		{
			name:  "vendor",
			files: map[string]string{"vendor/x.lgo": "package p\n"},
			want:  false,
		},
		{
			name:  "testdata",
			files: map[string]string{"testdata/x.lgo": "package p\n"},
			want:  false,
		},
		{
			name:  "dotDirectory",
			files: map[string]string{".hidden/x.lgo": "package p\n"},
			want:  false,
		},
		{
			name:  "underscoreDirectory",
			files: map[string]string{"_hidden/x.lgo": "package p\n"},
			want:  false,
		},
		{
			name: "nestedModuleFile",
			files: map[string]string{
				"sub/go.mod": "module m\n",
				"sub/x.lgo":  "package p\n",
			},
			want: false,
		},
		{
			name:  "goModDirectoryIsNotABoundary",
			dirs:  []string{"sub/go.mod"},
			files: map[string]string{"sub/x.lgo": "package p\n"},
			want:  true,
		},
		{
			name:  "rootFile",
			files: map[string]string{"x.lgo": "package p\n"},
			want:  true,
		},
		{
			name:  "nestedPackage",
			files: map[string]string{"pkg/x.lgo": "package p\n"},
			want:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := scanTree(t, tt.files, tt.dirs)
			if got := ModuleAt(root).Logos(); got != tt.want {
				t.Errorf("Logos() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestModuleAtParsesOnce(t *testing.T) {
	root := logosModule(t, "[build]\nopt_level = \"O3\"\n")
	before := moduleParseCount
	m1 := ModuleAt(root)
	m2 := ModuleAt(root)
	m3 := ModuleAt(root)
	if got := moduleParseCount - before; got != 1 {
		t.Fatalf("ModuleAt parsed the root %d times over 3 calls, want once", got)
	}
	if m1 != m2 || m2 != m3 {
		t.Fatal("ModuleAt returned different Module values for one root")
	}
	if m1.Err != nil || m1.File == nil || m1.File.Build.OptLevel != "O3" {
		t.Fatalf("Module = {err: %v, file: %v}, want the parsed file", m1.Err, m1.File)
	}
}

func TestModuleAtConcurrent(t *testing.T) {
	root := scanTree(t, map[string]string{"a.lgo": "package p\n"}, nil)
	before := moduleParseCount
	const workers = 32
	modules := make([]*Module, workers)
	var wg sync.WaitGroup
	for i := range workers {
		wg.Go(func() {
			modules[i] = ModuleAt(root)
			modules[i].Logos()
			modules[i].Severity("portability", "warn")
			modules[i].ClaimError()
		})
	}
	wg.Wait()
	if got := moduleParseCount - before; got != 1 {
		t.Fatalf("concurrent ModuleAt parsed the root %d times, want once", got)
	}
	for _, m := range modules {
		if m != modules[0] {
			t.Fatal("concurrent ModuleAt calls returned different Module values")
		}
		if !modules[0].Logos() {
			t.Fatal("Logos() = false, want true")
		}
	}
}

func TestClaimErrorOnce(t *testing.T) {
	root := logosModule(t, "[build]\narena = maybe\n")
	m := ModuleAt(root)
	if m.Err == nil {
		t.Fatal("Module.Err = nil, want the parse error")
	}
	if !m.ClaimError() {
		t.Fatal("first ClaimError = false, want true")
	}
	if m.ClaimError() {
		t.Fatal("second ClaimError = true, want false")
	}
	if again := ModuleAt(root); again.ClaimError() {
		t.Fatal("ClaimError fired again through the memoized module")
	}
}

func TestCgoAllowTexts(t *testing.T) {
	logosAllow := scanTree(t, map[string]string{
		"logos.toml": "[build]\ncgo = \"allow\"\n",
		"a.lgo":      "package p\n",
	}, nil)
	restrictedAllow := scanTree(t, map[string]string{
		"logos.toml": "[build]\ncgo = \"restricted\"\ncgo_allowlist = [\"example.com/ok\", \"example.com/ok/**\"]\n",
		"a.lgo":      "package p\n",
	}, nil)
	logosNoKey := scanTree(t, map[string]string{"a.lgo": "package p\n"}, nil)

	tests := []struct {
		name           string
		root           string
		cgoEnv         string
		importPath     string
		wantAllow      bool
		wantDenialText string
	}{
		{
			name: "allowKey", root: logosAllow, importPath: "example.com/m",
			wantAllow: true,
		},
		{
			name: "restrictedInAllowlist", root: restrictedAllow, importPath: "example.com/ok",
			wantAllow: true,
		},
		{
			name: "restrictedBelowAllowlist", root: restrictedAllow, importPath: "example.com/ok/sub/deep",
			wantAllow: true,
		},
		{
			name: "restrictedOutsideAllowlist", root: restrictedAllow, importPath: "example.com/m/other",
			wantDenialText: "[restriction::cgo_usage] package example.com/m/other uses cgo and is not in cgo_allowlist",
		},
		{
			name: "noKeyNoEnvLogosModule", root: logosNoKey, importPath: "example.com/m",
			wantDenialText: "[restriction::cgo_usage] cgo is disabled",
		},
		{
			name: "noKeyEnvOn", root: logosNoKey, cgoEnv: "1", importPath: "example.com/m",
			wantAllow: true,
		},
		{
			name: "noKeyEnvOff", root: logosNoKey, cgoEnv: "0", importPath: "example.com/m",
			wantDenialText: "[restriction::cgo_usage] cgo is disabled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allow, msg := ModuleAt(tt.root).CgoAllow(tt.cgoEnv, tt.importPath)
			if allow != tt.wantAllow {
				t.Errorf("CgoAllow allow = %v, want %v", allow, tt.wantAllow)
			}
			if tt.wantAllow {
				if msg != "" {
					t.Errorf("CgoAllow msg = %q, want empty on allow", msg)
				}
				return
			}
			if !strings.Contains(msg, "[restriction::cgo_usage]") {
				t.Errorf("CgoAllow msg = %q, want the [restriction::cgo_usage] denial", msg)
			}
			if tt.wantDenialText != "" && msg != tt.wantDenialText {
				t.Errorf("CgoAllow msg = %q, want %q", msg, tt.wantDenialText)
			}
		})
	}
}

func TestCgoAllowlistAllows(t *testing.T) {
	tests := []struct {
		name string
		list []string
		path string
		want bool
	}{
		{"exact", []string{"example.com/m/ok"}, "example.com/m/ok", true},
		{"exactMiss", []string{"example.com/m/ok"}, "example.com/m/other", false},
		{"starStarMatchesPrefix", []string{"example.com/m/ok/**"}, "example.com/m/ok", true},
		{"starStarMatchesBelow", []string{"example.com/m/ok/**"}, "example.com/m/ok/sub/deep", true},
		{"starStarSibling", []string{"example.com/m/ok/**"}, "example.com/m/okay", false},
		{"doubleStarSuffixAcrossSlash", []string{"example.com/a/o**"}, "example.com/a/okay/x", true},
		{"doubleStarSuffixSameDir", []string{"example.com/a/o**"}, "example.com/a/okay", true},
		{"doubleStarSuffixMiss", []string{"example.com/a/o**"}, "example.com/a/nope", false},
		{"bareDoubleStar", []string{"**"}, "example.com/a/b", true},
		{"starOneLevel", []string{"example.com/m/*"}, "example.com/m/a", true},
		{"starNoSlash", []string{"example.com/m/*"}, "example.com/m/a/b", false},
		{"question", []string{"example.com/m/o?"}, "example.com/m/ok", true},
	}
	for _, tt := range tests {
		if got := CgoAllowlistAllows(tt.list, tt.path); got != tt.want {
			t.Errorf("%s: CgoAllowlistAllows(%q, %q) = %v, want %v", tt.name, tt.list, tt.path, got, tt.want)
		}
	}
}

func TestParseRejectsInvalidCgoAllowlist(t *testing.T) {
	data := "[build]\ncgo = \"restricted\"\ncgo_allowlist = [\"[\"]\n"
	_, err := config.Parse("logos.toml", []byte(data))
	if err == nil {
		t.Fatal("Parse accepted invalid cgo_allowlist pattern \"[\", want [configuration::invalid_cgo_allowlist]")
	}
	const want = `[configuration::invalid_cgo_allowlist] cgo_allowlist pattern "[": syntax error in pattern`
	if !strings.Contains(err.Error(), want) {
		t.Errorf("Parse error = %q, want substring %q", err.Error(), want)
	}
}
