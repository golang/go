// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"context"
	"internal/testenv"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"

	"golang.org/x/mod/module"
)

func TestScanModuleKindsDuplicateSiblings(t *testing.T) {
	const modPath = "example.com/m"
	const duplicate = "correctness::duplicate_stem_filename"
	const mixedPackage = "correctness::mixed_package"
	tests := []struct {
		name       string
		files      map[string]string
		plainGo    bool
		selectedGo bool
		importErr  string
		scanErr    string
		wantGo     []string
		wantLogos  []string
	}{
		{
			name: "canonical_alias",
			files: map[string]string{
				"sibling/x.lgo":        "package sibling\n",
				"sibling/x.\U0001f4dc": "package sibling\n",
			},
			importErr: duplicate,
			wantLogos: []string{modPath + "/good", modPath + "/sibling"},
		},
		{
			name: "alias_alias",
			files: map[string]string{
				"sibling/x.\u26a1":     "package sibling\n",
				"sibling/x.\U0001f4dc": "package sibling\n",
			},
			importErr: duplicate,
			wantLogos: []string{modPath + "/good", modPath + "/sibling"},
		},
		{
			name: "variation_alias",
			files: map[string]string{
				"sibling/x.\u26a1":       "package sibling\n",
				"sibling/x.\u26a1\ufe0f": "package sibling\n",
			},
			importErr: duplicate,
			wantLogos: []string{modPath + "/good", modPath + "/sibling"},
		},
		{
			name: "external_test_alias",
			files: map[string]string{
				"sibling/x_test.lgo":        "package sibling_test\n",
				"sibling/x_test.\U0001f4dc": "package sibling_test\n",
			},
			importErr: duplicate,
			wantLogos: []string{modPath + "/good", modPath + "/sibling"},
		},
		{
			name: "duplicate_kind_still_mixes_module",
			files: map[string]string{
				"sibling/x.lgo":        "package sibling\n",
				"sibling/x.\U0001f4dc": "package sibling\n",
			},
			selectedGo: true,
			importErr:  duplicate,
			wantGo:     []string{modPath + "/good"},
			wantLogos:  []string{modPath + "/sibling"},
		},
		{
			name: "malformed_header_still_fails",
			files: map[string]string{
				"sibling/x.lgo":        "package\n",
				"sibling/x.\U0001f4dc": "package sibling\n",
			},
			importErr: "expected 'IDENT'",
			scanErr:   "expected 'IDENT'",
		},
		{
			name: "mixed_package_still_fails",
			files: map[string]string{
				"sibling/x.lgo":        "package sibling\n",
				"sibling/x.\U0001f4dc": "package sibling\n",
				"sibling/z.go":         "package sibling\n",
			},
			importErr: mixedPackage,
			scanErr:   mixedPackage,
		},
		{
			name: "plain_go_ignores_alias_siblings",
			files: map[string]string{
				"sibling/x.lgo":        "package sibling\n",
				"sibling/x.\U0001f4dc": "package sibling\n",
			},
			plainGo:    true,
			selectedGo: true,
			importErr:  "no buildable Go source files",
			wantGo:     []string{modPath + "/good"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if os.Getenv("GO_LOGOS_CENSUS_CHILD") != "1" {
				// A fresh process fixes the working directory before cfg, fsys,
				// module discovery, and cached build tags observe it.
				root := t.TempDir()
				files := maps.Clone(tt.files)
				files["go.mod"] = "module " + modPath + "\n\ngo 1.27\n"
				selected := "good/p.lgo"
				if tt.selectedGo {
					selected = "good/p.go"
				}
				files[selected] = "package good\n"
				for name, source := range files {
					path := filepath.Join(root, filepath.FromSlash(name))
					if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(path, []byte(source), 0o666); err != nil {
						t.Fatal(err)
					}
				}
				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				defer cancel()
				cmd := testenv.CommandContext(t, ctx, testenv.Executable(t), "-test.run=^TestScanModuleKindsDuplicateSiblings$/^"+tt.name+"$", "-test.v")
				cmd.Dir = root
				cmd.Env = append(cmd.Environ(),
					"GO_LOGOS_CENSUS_CHILD=1", "GO111MODULE=on", "GOWORK=off",
					"GOENV=off", "GOFLAGS=", "GOPROXY=off", "GOSUMDB=off", "GOTOOLCHAIN=local")
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("census subprocess: %v\n%s", err, out)
				}
				return
			}

			cfg.Logos = !tt.plainGo
			cfg.BuildContext.ToolTags = slices.DeleteFunc(slices.Clone(cfg.BuildContext.ToolTags), func(tag string) bool { return tag == "logos" })
			if cfg.Logos {
				cfg.BuildContext.ToolTags = append(cfg.BuildContext.ToolTags, "logos")
			}
			cfg.BuildContext.CgoEnabled = false
			ld := modload.NewLoader()
			modload.LoadModFile(ld, t.Context())
			if !ld.MainModules.Contains(modPath) {
				t.Fatal("test module was not loaded")
			}

			// Normal package selection must retain its local error. The census
			// may ignore only a duplicate-stem error with complete file kinds.
			if _, err := cfg.BuildContext.ImportDir("sibling", 0); err == nil || !strings.Contains(err.Error(), tt.importErr) {
				t.Fatalf("selected sibling error = %v, want %q", err, tt.importErr)
			}
			g := &moduleKinds{
				goPkgs:    make(map[string]bool),
				logosPkgs: make(map[string]bool),
				module:    module.Version{Path: modPath},
				carrier:   &Package{PackagePublic: PackagePublic{ImportPath: modPath + "/good"}},
			}
			if tt.selectedGo {
				g.goPkgs[modPath+"/good"] = true
			} else {
				g.logosPkgs[modPath+"/good"] = true
			}
			err := scanModuleKinds(t.Context(), ld, g)
			if tt.scanErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.scanErr) || !strings.Contains(err.Error(), "classifying package "+modPath+"/sibling in module "+modPath+":") {
					t.Fatalf("scanModuleKinds error = %v, want sibling classification error containing %q", err, tt.scanErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("scanModuleKinds rejected a classifiable sibling: %v", err)
			}
			if got := slices.Sorted(maps.Keys(g.goPkgs)); !slices.Equal(got, tt.wantGo) {
				t.Errorf("Go packages = %q, want %q", got, tt.wantGo)
			}
			if got := slices.Sorted(maps.Keys(g.logosPkgs)); !slices.Equal(got, tt.wantLogos) {
				t.Errorf("loGo/s packages = %q, want %q", got, tt.wantLogos)
			}
			reportMixedModule(modPath, g)
			if len(tt.wantGo) != 0 && len(tt.wantLogos) != 0 {
				if g.carrier.Error == nil || !g.carrier.Incomplete || !strings.Contains(g.carrier.Error.Error(), "correctness::mixed_module") {
					t.Fatalf("mixed module error = %v, incomplete = %v; want mixed_module rejection", g.carrier.Error, g.carrier.Incomplete)
				}
			} else if g.carrier.Error != nil || g.carrier.Incomplete {
				t.Fatalf("pure module error = %v, incomplete = %v; want no error", g.carrier.Error, g.carrier.Incomplete)
			}
		})
	}
}
