// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"os"
	"path/filepath"

	"golang.org/x/mod/modfile"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/settings"
	"cmd/go/internal/modload"
)

var logosSettingsErr error

func refreshLogosPolicy(ld *modload.Loader) {
	f := settings.CommandLine
	f.Logos, f.Env, f.Roots = cfg.Logos, cfg.LogosStartupEnv(), logosMainRoots(ld)
	s, err := settings.Load(settings.CWDModuleRoot(), f)
	settings.SetCurrent(s)
	logosSettingsErr = err
	cfg.ApplyLogosEnv(s)
}

func logosMainRoots(ld *modload.Loader) []string {
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
	return roots
}

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
