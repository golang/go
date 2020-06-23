// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/par"
	"cmd/go/internal/search"

	"golang.org/x/mod/module"
)

func ListModules(args []string, listU, listVersions bool) []*modinfo.ModulePublic {
	mods := listModules(args, listVersions)
	if listU || listVersions {
		var work par.Work
		for _, m := range mods {
			work.Add(m)
			if m.Replace != nil {
				work.Add(m.Replace)
			}
		}
		work.Do(10, func(item interface{}) {
			m := item.(*modinfo.ModulePublic)
			if listU {
				addUpdate(m)
			}
			if listVersions {
				addVersions(m)
			}
		})
	}
	return mods
}

func listModules(args []string, listVersions bool) []*modinfo.ModulePublic {
	LoadBuildList()
	if len(args) == 0 {
		return []*modinfo.ModulePublic{moduleInfo(buildList[0], true)}
	}

	var mods []*modinfo.ModulePublic
	matchedBuildList := make([]bool, len(buildList))
	for _, arg := range args {
		if strings.Contains(arg, `\`) {
			base.Fatalf("go: module paths never use backslash")
		}
		if search.IsRelativePath(arg) {
			base.Fatalf("go: cannot use relative path %s to specify module", arg)
		}
		if !HasModRoot() && (arg == "all" || strings.Contains(arg, "...")) {
			base.Fatalf("go: cannot match %q: working directory is not part of a module", arg)
		}
		if i := strings.Index(arg, "@"); i >= 0 {
			path := arg[:i]
			vers := arg[i+1:]
			var current string
			for _, m := range buildList {
				if m.Path == path {
					current = m.Version
					break
				}
			}

			info, err := Query(path, vers, current, nil)
			if err != nil {
				mods = append(mods, &modinfo.ModulePublic{
					Path:    path,
					Version: vers,
					Error:   modinfoError(path, vers, err),
				})
				continue
			}
			mods = append(mods, moduleInfo(module.Version{Path: path, Version: info.Version}, false))
			continue
		}

		// Module path or pattern.
		var match func(string) bool
		var literal bool
		if arg == "all" {
			match = func(string) bool { return true }
		} else if strings.Contains(arg, "...") {
			match = search.MatchPattern(arg)
		} else {
			match = func(p string) bool { return arg == p }
			literal = true
		}
		matched := false
		for i, m := range buildList {
			if i == 0 && !HasModRoot() {
				// The root module doesn't actually exist: omit it.
				continue
			}
			if match(m.Path) {
				matched = true
				if !matchedBuildList[i] {
					matchedBuildList[i] = true
					mods = append(mods, moduleInfo(m, true))
				}
			}
		}
		if !matched {
			if literal {
				if listVersions {
					// Don't make the user provide an explicit '@latest' when they're
					// explicitly asking what the available versions are.
					// Instead, resolve the module, even if it isn't an existing dependency.
					info, err := Query(arg, "latest", "", nil)
					if err == nil {
						mods = append(mods, moduleInfo(module.Version{Path: arg, Version: info.Version}, false))
					} else {
						mods = append(mods, &modinfo.ModulePublic{
							Path:  arg,
							Error: modinfoError(arg, "", err),
						})
					}
					continue
				}
				if cfg.BuildMod == "vendor" {
					// In vendor mode, we can't determine whether a missing module is “a
					// known dependency” because the module graph is incomplete.
					// Give a more explicit error message.
					mods = append(mods, &modinfo.ModulePublic{
						Path:  arg,
						Error: modinfoError(arg, "", errors.New("can't resolve module using the vendor directory\n\t(Use -mod=mod or -mod=readonly to bypass.)")),
					})
				} else {
					mods = append(mods, &modinfo.ModulePublic{
						Path:  arg,
						Error: modinfoError(arg, "", errors.New("not a known dependency")),
					})
				}
			} else {
				fmt.Fprintf(os.Stderr, "warning: pattern %q matched no module dependencies\n", arg)
			}
		}
	}

	return mods
}

// modinfoError wraps an error to create an error message in
// modinfo.ModuleError with minimal redundancy.
func modinfoError(path, vers string, err error) *modinfo.ModuleError {
	var nerr *NoMatchingVersionError
	var merr *module.ModuleError
	if errors.As(err, &nerr) {
		// NoMatchingVersionError contains the query, so we don't mention the
		// query again in ModuleError.
		err = &module.ModuleError{Path: path, Err: err}
	} else if !errors.As(err, &merr) {
		// If the error does not contain path and version, wrap it in a
		// module.ModuleError.
		err = &module.ModuleError{Path: path, Version: vers, Err: err}
	}

	return &modinfo.ModuleError{Err: err.Error()}
}
