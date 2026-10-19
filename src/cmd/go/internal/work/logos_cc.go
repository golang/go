// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"cmp"
	"os"
	"path/filepath"

	"golang.org/x/mod/modfile"

	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
)

// refreshLogosCgoPolicy recomputes the global cgo switch from every main
// module root once flags are parsed and modload has located the workspace
// or module. ld.MainModules is still unpopulated this early (it is built
// lazily during package loading), so in workspace mode the member list
// comes from the go.work use directives instead; with no workspace the
// module above the working directory decides via cfg. It runs in BuildInit,
// before package loading decides cgo file visibility, so -C targets and
// mixed-policy workspaces apply the owning modules' policies instead of
// the startup directory's.
func refreshLogosCgoPolicy(ld *modload.Loader) {
	var roots []string
	if ld != nil {
		if ld.MainModules != nil {
			for _, m := range ld.MainModules.Versions() {
				if root := ld.MainModules.ModRoot(m); root != "" {
					roots = append(roots, root)
				}
			}
		}
		if len(roots) == 0 {
			roots = workspaceMemberRoots(modload.WorkFilePath(ld))
		}
	}
	cfg.RefreshLogosCgoPolicy(roots)
}

// workspaceMemberRoots lists the member directories of the workspace file
// at workFile, or nil outside workspace mode. go.work use paths resolve
// against the work file's directory; absolute paths are kept as is. An
// empty, missing, or unparsable file yields nil through the read and parse
// guards so callers fall back to the working-directory module.
func workspaceMemberRoots(workFile string) []string {
	data, err := os.ReadFile(workFile)
	if err != nil {
		return nil
	}
	wf, err := modfile.ParseWork(workFile, data, nil)
	if err != nil {
		return nil
	}
	dir := filepath.Dir(workFile)
	var roots []string
	for _, u := range wf.Use {
		p := u.Path
		if !filepath.IsAbs(p) {
			p = filepath.Join(dir, filepath.FromSlash(p))
		}
		roots = append(roots, p)
	}
	return roots
}

// logosCompilerFallback resolves the CC/CXX compiler for a cgo build
// through the cfg zig fallback ("zig cc -target <triple>" or
// "zig c++ -target <triple>"), which honors the startup snapshots. Plain
// envList callers keep their behavior for every other key.
func logosCompilerFallback(key, def string) string {
	if key == "CXX" {
		return cfg.ZigCXX(def)
	}
	return cfg.ZigCC(def)
}

// logosCgoToolEnv returns CC/CXX overrides for the cgo tool's environment
// carrying the resolved zig fallback commands, or nil when the fallback is
// inactive. The cgo tool resolves its own compiler from its $CC (see
// checkGCCBaseCmd in cmd/cgo), ignoring the Go-side envList resolution, so
// without these entries it rejects a hidden default compiler before the
// fallback can run. Callers append the result after the base environment so
// it wins over the MkEnv-injected CC default; an explicit user CC/CXX (or
// a present default compiler, or disabled cgo) resolves unchanged and
// yields no override, and "go env CC" output is untouched.
func logosCgoToolEnv() []string {
	var env []string
	cc := cmp.Or(cfg.Getenv("CC"), cfg.DefaultCC(cfg.Goos, cfg.Goarch))
	if fb := cfg.ZigCC(cc); fb != cc {
		env = append(env, "CC="+fb)
	}
	cxx := cmp.Or(cfg.Getenv("CXX"), cfg.DefaultCXX(cfg.Goos, cfg.Goarch))
	if fb := cfg.ZigCXX(cxx); fb != cxx {
		env = append(env, "CXX="+fb)
	}
	return env
}
