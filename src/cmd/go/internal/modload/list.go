// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/search"

	"golang.org/x/mod/module"
)

type ListMode int

const (
	ListU ListMode = 1 << iota
	ListRetracted
	ListDeprecated
	ListVersions
	ListRetractedVersions
)

// ListModules returns a description of the modules matching args, if known,
// along with any error preventing additional matches from being identified.
//
// The returned slice can be nonempty even if the error is non-nil.
func ListModules(ctx context.Context, args []string, mode ListMode) ([]*modinfo.ModulePublic, error) {
	rs, mods, err := listModules(ctx, LoadModFile(ctx), args, mode)

	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))
	if mode != 0 {
		for _, m := range mods {
			add := func(m *modinfo.ModulePublic) {
				sem <- token{}
				go func() {
					if mode&ListU != 0 {
						addUpdate(ctx, m)
					}
					if mode&ListVersions != 0 {
						addVersions(ctx, m, mode&ListRetractedVersions != 0)
					}
					if mode&ListRetracted != 0 {
						addRetraction(ctx, m)
					}
					if mode&ListDeprecated != 0 {
						addDeprecation(ctx, m)
					}
					<-sem
				}()
			}

			add(m)
			if m.Replace != nil {
				add(m.Replace)
			}
		}
	}
	// Fill semaphore channel to wait for all tasks to finish.
	for n := cap(sem); n > 0; n-- {
		sem <- token{}
	}

	if err == nil {
		commitRequirements(ctx, modFileGoVersion(), rs)
	}
	return mods, err
}

func listModules(ctx context.Context, rs *Requirements, args []string, mode ListMode) (_ *Requirements, mods []*modinfo.ModulePublic, mgErr error) {
	if len(args) == 0 {
		return rs, []*modinfo.ModulePublic{moduleInfo(ctx, rs, Target, mode)}, nil
	}

	needFullGraph := false
	for _, arg := range args {
		if strings.Contains(arg, `\`) {
			base.Fatalf("go: module paths never use backslash")
		}
		if search.IsRelativePath(arg) {
			base.Fatalf("go: cannot use relative path %s to specify module", arg)
		}
		if arg == "all" || strings.Contains(arg, "...") {
			needFullGraph = true
			if !HasModRoot() {
				base.Fatalf("go: cannot match %q: %v", arg, ErrNoModRoot)
			}
			continue
		}
		if i := strings.Index(arg, "@"); i >= 0 {
			path := arg[:i]
			vers := arg[i+1:]
			if vers == "upgrade" || vers == "patch" {
				if _, ok := rs.rootSelected(path); !ok || rs.depth == eager {
					needFullGraph = true
					if !HasModRoot() {
						base.Fatalf("go: cannot match %q: %v", arg, ErrNoModRoot)
					}
				}
			}
			continue
		}
		if _, ok := rs.rootSelected(arg); !ok || rs.depth == eager {
			needFullGraph = true
			if mode&ListVersions == 0 && !HasModRoot() {
				base.Fatalf("go: cannot match %q without -versions or an explicit version: %v", arg, ErrNoModRoot)
			}
		}
	}

	var mg *ModuleGraph
	if needFullGraph {
		rs, mg, mgErr = expandGraph(ctx, rs)
	}

	matchedModule := map[module.Version]bool{}
	for _, arg := range args {
		if i := strings.Index(arg, "@"); i >= 0 {
			path := arg[:i]
			vers := arg[i+1:]

			var current string
			if mg == nil {
				current, _ = rs.rootSelected(path)
			} else {
				current = mg.Selected(path)
			}
			if current == "none" && mgErr != nil {
				if vers == "upgrade" || vers == "patch" {
					// The module graph is incomplete, so we don't know what version we're
					// actually upgrading from.
					// mgErr is already set, so just skip this module.
					continue
				}
			}

			allowed := CheckAllowed
			if IsRevisionQuery(vers) || mode&ListRetracted != 0 {
				// Allow excluded and retracted versions if the user asked for a
				// specific revision or used 'go list -retracted'.
				allowed = nil
			}
			info, err := Query(ctx, path, vers, current, allowed)
			if err != nil {
				mods = append(mods, &modinfo.ModulePublic{
					Path:    path,
					Version: vers,
					Error:   modinfoError(path, vers, err),
				})
				continue
			}

			// Indicate that m was resolved from outside of rs by passing a nil
			// *Requirements instead.
			var noRS *Requirements

			mod := moduleInfo(ctx, noRS, module.Version{Path: path, Version: info.Version}, mode)
			mods = append(mods, mod)
			continue
		}

		// Module path or pattern.
		var match func(string) bool
		if arg == "all" {
			match = func(string) bool { return true }
		} else if strings.Contains(arg, "...") {
			match = search.MatchPattern(arg)
		} else {
			var v string
			if mg == nil {
				var ok bool
				v, ok = rs.rootSelected(arg)
				if !ok {
					// We checked rootSelected(arg) in the earlier args loop, so if there
					// is no such root we should have loaded a non-nil mg.
					panic(fmt.Sprintf("internal error: root requirement expected but not found for %v", arg))
				}
			} else {
				v = mg.Selected(arg)
			}
			if v == "none" && mgErr != nil {
				// mgErr is already set, so just skip this module.
				continue
			}
			if v != "none" {
				mods = append(mods, moduleInfo(ctx, rs, module.Version{Path: arg, Version: v}, mode))
			} else if cfg.BuildMod == "vendor" {
				// In vendor mode, we can't determine whether a missing module is “a
				// known dependency” because the module graph is incomplete.
				// Give a more explicit error message.
				mods = append(mods, &modinfo.ModulePublic{
					Path:  arg,
					Error: modinfoError(arg, "", errors.New("can't resolve module using the vendor directory\n\t(Use -mod=mod or -mod=readonly to bypass.)")),
				})
			} else if mode&ListVersions != 0 {
				// Don't make the user provide an explicit '@latest' when they're
				// explicitly asking what the available versions are. Instead, return a
				// module with version "none", to which we can add the requested list.
				mods = append(mods, &modinfo.ModulePublic{Path: arg})
			} else {
				mods = append(mods, &modinfo.ModulePublic{
					Path:  arg,
					Error: modinfoError(arg, "", errors.New("not a known dependency")),
				})
			}
			continue
		}

		matched := false
		for _, m := range mg.BuildList() {
			if match(m.Path) {
				matched = true
				if !matchedModule[m] {
					matchedModule[m] = true
					mods = append(mods, moduleInfo(ctx, rs, m, mode))
				}
			}
		}
		if !matched {
			fmt.Fprintf(os.Stderr, "warning: pattern %q matched no module dependencies\n", arg)
		}
	}

	return rs, mods, mgErr
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
