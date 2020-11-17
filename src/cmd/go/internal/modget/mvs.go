// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modget

import (
	"context"
	"errors"

	"cmd/go/internal/base"
	"cmd/go/internal/modload"
	"cmd/go/internal/mvs"

	"golang.org/x/mod/module"
)

// An upgrader adapts an underlying mvs.Reqs to apply an
// upgrade policy to a list of targets and their dependencies.
type upgrader struct {
	mvs.Reqs

	// cmdline maps a module path to a query made for that module at a
	// specific target version. Each query corresponds to a module
	// matched by a command line argument.
	cmdline map[string]*query

	// upgrade is a set of modules providing dependencies of packages
	// matched by command line arguments. If -u or -u=patch is set,
	// these modules are upgraded accordingly.
	upgrade map[string]bool
}

// newUpgrader creates an upgrader. cmdline contains queries made at
// specific versions for modules matched by command line arguments. pkgs
// is the set of packages matched by command line arguments. If -u or -u=patch
// is set, modules providing dependencies of pkgs are upgraded accordingly.
func newUpgrader(cmdline map[string]*query, pkgs map[string]bool) *upgrader {
	u := &upgrader{
		Reqs:    modload.Reqs(),
		cmdline: cmdline,
	}
	if getU != "" {
		u.upgrade = make(map[string]bool)

		// Traverse package import graph.
		// Initialize work queue with root packages.
		seen := make(map[string]bool)
		var work []string
		add := func(path string) {
			if !seen[path] {
				seen[path] = true
				work = append(work, path)
			}
		}
		for pkg := range pkgs {
			add(pkg)
		}
		for len(work) > 0 {
			pkg := work[0]
			work = work[1:]
			m := modload.PackageModule(pkg)
			u.upgrade[m.Path] = true

			// testImports is empty unless test imports were actually loaded,
			// i.e., -t was set or "all" was one of the arguments.
			imports, testImports := modload.PackageImports(pkg)
			for _, imp := range imports {
				add(imp)
			}
			for _, imp := range testImports {
				add(imp)
			}
		}
	}
	return u
}

// Required returns the requirement list for m.
// For the main module, we override requirements with the modules named
// one the command line, and we include new requirements. Otherwise,
// we defer to u.Reqs.
func (u *upgrader) Required(m module.Version) ([]module.Version, error) {
	rs, err := u.Reqs.Required(m)
	if err != nil {
		return nil, err
	}
	if m != modload.Target {
		return rs, nil
	}

	overridden := make(map[string]bool)
	for i, m := range rs {
		if q := u.cmdline[m.Path]; q != nil && q.m.Version != "none" {
			rs[i] = q.m
			overridden[q.m.Path] = true
		}
	}
	for _, q := range u.cmdline {
		if !overridden[q.m.Path] && q.m.Path != modload.Target.Path && q.m.Version != "none" {
			rs = append(rs, q.m)
		}
	}
	return rs, nil
}

// Upgrade returns the desired upgrade for m.
//
// If m was requested at a specific version on the command line, then
// Upgrade returns that version.
//
// If -u is set and m provides a dependency of a package matched by
// command line arguments, then Upgrade may provider a newer tagged version.
// If m is a tagged version, then Upgrade will return the latest tagged
// version (with the same minor version number if -u=patch).
// If m is a pseudo-version, then Upgrade returns the latest tagged version
// only if that version has a time-stamp newer than m. This special case
// prevents accidental downgrades when already using a pseudo-version
// newer than the latest tagged version.
//
// If none of the above cases apply, then Upgrade returns m.
func (u *upgrader) Upgrade(m module.Version) (module.Version, error) {
	// Allow pkg@vers on the command line to override the upgrade choice v.
	// If q's version is < m.Version, then we're going to downgrade anyway,
	// and it's cleaner to avoid moving back and forth and picking up
	// extraneous other newer dependencies.
	// If q's version is > m.Version, then we're going to upgrade past
	// m.Version anyway, and again it's cleaner to avoid moving back and forth
	// picking up extraneous other newer dependencies.
	if q := u.cmdline[m.Path]; q != nil {
		return q.m, nil
	}

	if !u.upgrade[m.Path] {
		// Not involved in upgrade. Leave alone.
		return m, nil
	}

	// Run query required by upgrade semantics.
	// Note that Query "latest" is not the same as using repo.Latest,
	// which may return a pseudoversion for the latest commit.
	// Query "latest" returns the newest tagged version or the newest
	// prerelease version if there are no non-prereleases, or repo.Latest
	// if there aren't any tagged versions.
	// If we're querying "upgrade" or "patch", Query will compare the current
	// version against the chosen version and will return the current version
	// if it is newer.
	info, err := modload.Query(context.TODO(), m.Path, string(getU), m.Version, modload.CheckAllowed)
	if err != nil {
		// Report error but return m, to let version selection continue.
		// (Reporting the error will fail the command at the next base.ExitIfErrors.)

		// Special case: if the error is for m.Version itself and m.Version has a
		// replacement, then keep it and don't report the error: the fact that the
		// version is invalid is likely the reason it was replaced to begin with.
		var vErr *module.InvalidVersionError
		if errors.As(err, &vErr) && vErr.Version == m.Version && modload.Replacement(m).Path != "" {
			return m, nil
		}

		// Special case: if the error is "no matching versions" then don't
		// even report the error. Because Query does not consider pseudo-versions,
		// it may happen that we have a pseudo-version but during -u=patch
		// the query v0.0 matches no versions (not even the one we're using).
		var noMatch *modload.NoMatchingVersionError
		if !errors.As(err, &noMatch) {
			base.Errorf("go get: upgrading %s@%s: %v", m.Path, m.Version, err)
		}
		return m, nil
	}

	if info.Version != m.Version {
		logOncef("go: %s %s => %s", m.Path, getU, info.Version)
	}
	return module.Version{Path: m.Path, Version: info.Version}, nil
}

// buildListForLostUpgrade returns the build list for the module graph
// rooted at lost. Unlike mvs.BuildList, the target module (lost) is not
// treated specially. The returned build list may contain a newer version
// of lost.
//
// buildListForLostUpgrade is used after a downgrade has removed a module
// requested at a specific version. This helps us understand the requirements
// implied by each downgrade.
func buildListForLostUpgrade(lost module.Version, reqs mvs.Reqs) ([]module.Version, error) {
	return mvs.BuildList(lostUpgradeRoot, &lostUpgradeReqs{Reqs: reqs, lost: lost})
}

var lostUpgradeRoot = module.Version{Path: "lost-upgrade-root", Version: ""}

type lostUpgradeReqs struct {
	mvs.Reqs
	lost module.Version
}

func (r *lostUpgradeReqs) Required(mod module.Version) ([]module.Version, error) {
	if mod == lostUpgradeRoot {
		return []module.Version{r.lost}, nil
	}
	return r.Reqs.Required(mod)
}
