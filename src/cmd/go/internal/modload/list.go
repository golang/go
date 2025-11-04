// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/search"
	"cmd/internal/par"
	"cmd/internal/pkgpattern"

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
func ListModules(loaderstate *State, ctx context.Context, args []string, mode ListMode, reuseFile string) ([]*modinfo.ModulePublic, error) {
	var reuse map[module.Version]*modinfo.ModulePublic
	if reuseFile != "" {
		data, err := os.ReadFile(reuseFile)
		if err != nil {
			return nil, err
		}
		dec := json.NewDecoder(bytes.NewReader(data))
		reuse = make(map[module.Version]*modinfo.ModulePublic)
		for {
			var m modinfo.ModulePublic
			if err := dec.Decode(&m); err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("parsing %s: %v", reuseFile, err)
			}
			if m.Origin == nil {
				continue
			}
			m.Reuse = true
			reuse[module.Version{Path: m.Path, Version: m.Version}] = &m
			if m.Query != "" {
				reuse[module.Version{Path: m.Path, Version: m.Query}] = &m
			}
		}
	}

	rs, mods, err := listModules(loaderstate, ctx, LoadModFile(loaderstate, ctx), args, mode, reuse)

	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))
	if mode != 0 {
		for _, m := range mods {
			if m.Reuse {
				continue
			}
			add := func(m *modinfo.ModulePublic) {
				sem <- token{}
				go func() {
					if mode&ListU != 0 {
						addUpdate(loaderstate, ctx, m)
					}
					if mode&ListVersions != 0 {
						addVersions(loaderstate, ctx, m, mode&ListRetractedVersions != 0)
					}
					if mode&ListRetracted != 0 {
						addRetraction(loaderstate, ctx, m)
					}
					if mode&ListDeprecated != 0 {
						addDeprecation(loaderstate, ctx, m)
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
		loaderstate.requirements = rs
		// TODO(#61605): The extra ListU clause fixes a problem with Go 1.21rc3
		// where "go mod tidy" and "go list -m -u all" fight over whether the go.sum
		// should be considered up-to-date. The fix for now is to always treat the
		// go.sum as up-to-date during list -m -u. Probably the right fix is more targeted,
		// but in general list -u is looking up other checksums in the checksum database
		// that won't be necessary later, so it makes sense not to write the go.sum back out.
		if !ExplicitWriteGoMod && mode&ListU == 0 {
			err = commitRequirements(loaderstate, ctx, WriteOpts{})
		}
	}
	return mods, err
}

func listModules(loaderstate *State, ctx context.Context, rs *Requirements, args []string, mode ListMode, reuse map[module.Version]*modinfo.ModulePublic) (_ *Requirements, mods []*modinfo.ModulePublic, mgErr error) {
	if len(args) == 0 {
		var ms []*modinfo.ModulePublic
		for _, m := range loaderstate.MainModules.Versions() {
			if gover.IsToolchain(m.Path) {
				continue
			}
			ms = append(ms, moduleInfo(loaderstate, ctx, rs, m, mode, reuse))
		}
		return rs, ms, nil
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
			if !loaderstate.HasModRoot() {
				base.Fatalf("go: cannot match %q: %v", arg, NewNoMainModulesError(loaderstate))
			}
			continue
		}
		if path, vers, found := strings.Cut(arg, "@"); found {
			if vers == "upgrade" || vers == "patch" {
				if _, ok := rs.rootSelected(loaderstate, path); !ok || rs.pruning == unpruned {
					needFullGraph = true
					if !loaderstate.HasModRoot() {
						base.Fatalf("go: cannot match %q: %v", arg, NewNoMainModulesError(loaderstate))
					}
				}
			}
			continue
		}
		if _, ok := rs.rootSelected(loaderstate, arg); !ok || rs.pruning == unpruned {
			needFullGraph = true
			if mode&ListVersions == 0 && !loaderstate.HasModRoot() {
				base.Fatalf("go: cannot match %q without -versions or an explicit version: %v", arg, NewNoMainModulesError(loaderstate))
			}
		}
	}

	var mg *ModuleGraph
	if needFullGraph {
		rs, mg, mgErr = expandGraph(loaderstate, ctx, rs)
	}

	matchedModule := map[module.Version]bool{}
	for _, arg := range args {
		if path, vers, found := strings.Cut(arg, "@"); found {
			var current string
			if mg == nil {
				current, _ = rs.rootSelected(loaderstate, path)
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

			allowed := loaderstate.CheckAllowed
			if IsRevisionQuery(path, vers) || mode&ListRetracted != 0 {
				// Allow excluded and retracted versions if the user asked for a
				// specific revision or used 'go list -retracted'.
				allowed = nil
			}
			info, err := queryReuse(loaderstate, ctx, path, vers, current, allowed, reuse)
			if err != nil {
				var origin *codehost.Origin
				if info != nil {
					origin = info.Origin
				}
				mods = append(mods, &modinfo.ModulePublic{
					Path:    path,
					Version: vers,
					Error:   modinfoError(path, vers, err),
					Origin:  origin,
				})
				continue
			}

			// Indicate that m was resolved from outside of rs by passing a nil
			// *Requirements instead.
			var noRS *Requirements

			mod := moduleInfo(loaderstate, ctx, noRS, module.Version{Path: path, Version: info.Version}, mode, reuse)
			if vers != mod.Version {
				mod.Query = vers
			}
			mod.Origin = info.Origin
			mods = append(mods, mod)
			continue
		}

		// Module path or pattern.
		var match func(string) bool
		if arg == "all" {
			match = func(p string) bool { return !gover.IsToolchain(p) }
		} else if strings.Contains(arg, "...") {
			mp := pkgpattern.MatchPattern(arg)
			match = func(p string) bool { return mp(p) && !gover.IsToolchain(p) }
		} else {
			var v string
			if mg == nil {
				var ok bool
				v, ok = rs.rootSelected(loaderstate, arg)
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
				mods = append(mods, moduleInfo(loaderstate, ctx, rs, module.Version{Path: arg, Version: v}, mode, reuse))
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

		var matches []module.Version
		for _, m := range mg.BuildList() {
			if match(m.Path) {
				if !matchedModule[m] {
					matchedModule[m] = true
					matches = append(matches, m)
				}
			}
		}

		if len(matches) == 0 {
			fmt.Fprintf(os.Stderr, "warning: pattern %q matched no module dependencies\n", arg)
		}

		q := par.NewQueue(runtime.GOMAXPROCS(0))
		fetchedMods := make([]*modinfo.ModulePublic, len(matches))
		for i, m := range matches {
			q.Add(func() {
				fetchedMods[i] = moduleInfo(loaderstate, ctx, rs, m, mode, reuse)
			})
		}
		<-q.Idle()
		mods = append(mods, fetchedMods...)
	}

	return rs, mods, mgErr
}

// modinfoError wraps an error to create an error message in
// modinfo.ModuleError with minimal redundancy.
func modinfoError(path, vers string, err error) *modinfo.ModuleError {
	if _, ok := errors.AsType[*NoMatchingVersionError](err); ok {
		// NoMatchingVersionError contains the query, so we don't mention the
		// query again in ModuleError.
		err = &module.ModuleError{Path: path, Err: err}
	} else if _, ok := errors.AsType[*module.ModuleError](err); !ok {
		// If the error does not contain path and version, wrap it in a
		// module.ModuleError.
		err = &module.ModuleError{Path: path, Version: vers, Err: err}
	}

	return &modinfo.ModuleError{Err: err.Error()}
}
