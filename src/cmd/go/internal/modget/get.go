// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package modget implements the module-aware ``go get'' command.
package modget

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/get"
	"cmd/go/internal/load"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"
	"cmd/go/internal/module"
	"cmd/go/internal/mvs"
	"cmd/go/internal/par"
	"cmd/go/internal/search"
	"cmd/go/internal/semver"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
	"fmt"
	"os"
	pathpkg "path"
	"path/filepath"
	"strings"
)

var CmdGet = &base.Command{
	// Note: -d -m -u are listed explicitly because they are the most common get flags.
	// Do not send CLs removing them because they're covered by [get flags].
	UsageLine: "get [-d] [-m] [-u] [-v] [-insecure] [build flags] [packages]",
	Short:     "add dependencies to current module and install them",
	Long: `
Get resolves and adds dependencies to the current development module
and then builds and installs them.

The first step is to resolve which dependencies to add. 

For each named package or package pattern, get must decide which version of
the corresponding module to use. By default, get chooses the latest tagged
release version, such as v0.4.5 or v1.2.3. If there are no tagged release
versions, get chooses the latest tagged prerelease version, such as
v0.0.1-pre1. If there are no tagged versions at all, get chooses the latest
known commit.

This default version selection can be overridden by adding an @version
suffix to the package argument, as in 'go get golang.org/x/text@v0.3.0'.
For modules stored in source control repositories, the version suffix can
also be a commit hash, branch identifier, or other syntax known to the
source control system, as in 'go get golang.org/x/text@master'.
The version suffix @latest explicitly requests the default behavior
described above.

If a module under consideration is already a dependency of the current
development module, then get will update the required version.
Specifying a version earlier than the current required version is valid and
downgrades the dependency. The version suffix @none indicates that the
dependency should be removed entirely.

Although get defaults to using the latest version of the module containing
a named package, it does not use the latest version of that module's
dependencies. Instead it prefers to use the specific dependency versions
requested by that module. For example, if the latest A requires module
B v1.2.3, while B v1.2.4 and v1.3.1 are also available, then 'go get A'
will use the latest A but then use B v1.2.3, as requested by A. (If there
are competing requirements for a particular module, then 'go get' resolves
those requirements by taking the maximum requested version.)

The -u flag instructs get to update dependencies to use newer minor or
patch releases when available. Continuing the previous example,
'go get -u A' will use the latest A with B v1.3.1 (not B v1.2.3).

The -u=patch flag (not -u patch) instructs get to update dependencies
to use newer patch releases when available. Continuing the previous example,
'go get -u=patch A' will use the latest A with B v1.2.4 (not B v1.2.3).

In general, adding a new dependency may require upgrading
existing dependencies to keep a working build, and 'go get' does
this automatically. Similarly, downgrading one dependency may
require downgrading other dependenceis, and 'go get' does
this automatically as well.

The -m flag instructs get to stop here, after resolving, upgrading,
and downgrading modules and updating go.mod. When using -m,
each specified package path must be a module path as well,
not the import path of a package below the module root.

The -insecure flag permits fetching from repositories and resolving
custom domains using insecure schemes such as HTTP. Use with caution.

The second step is to download (if needed), build, and install
the named packages.

If an argument names a module but not a package (because there is no
Go source code in the module's root directory), then the install step
is skipped for that argument, instead of causing a build failure.
For example 'go get golang.org/x/perf' succeeds even though there
is no code corresponding to that import path.

Note that package patterns are allowed and are expanded after resolving
the module versions. For example, 'go get golang.org/x/perf/cmd/...'
adds the latest golang.org/x/perf and then installs the commands in that
latest version.

The -d flag instructs get to download the source code needed to build
the named packages, including downloading necessary dependencies,
but not to build and install them.

With no package arguments, 'go get' applies to the main module,
and to the Go package in the current directory, if any. In particular,
'go get -u' and 'go get -u=patch' update all the dependencies of the
main module. With no package arguments and also without -u,
'go get' is not much more than 'go install', and 'go get -d' not much
more than 'go list'.

For more about modules, see 'go help modules'.

For more about specifying packages, see 'go help packages'.

This text describes the behavior of get using modules to manage source
code and dependencies. If instead the go command is running in GOPATH
mode, the details of get's flags and effects change, as does 'go help get'.
See 'go help modules' and 'go help gopath-get'.

See also: go build, go install, go clean, go mod.
	`,
}

// Note that this help text is a stopgap to make the module-aware get help text
// available even in non-module settings. It should be deleted when the old get
// is deleted. It should NOT be considered to set a precedent of having hierarchical
// help names with dashes.
var HelpModuleGet = &base.Command{
	UsageLine: "module-get",
	Short:     "module-aware go get",
	Long: `
The 'go get' command changes behavior depending on whether the
go command is running in module-aware mode or legacy GOPATH mode.
This help text, accessible as 'go help module-get' even in legacy GOPATH mode,
describes 'go get' as it operates in module-aware mode.

Usage: ` + CmdGet.UsageLine + `
` + CmdGet.Long,
}

var (
	getD   = CmdGet.Flag.Bool("d", false, "")
	getF   = CmdGet.Flag.Bool("f", false, "")
	getFix = CmdGet.Flag.Bool("fix", false, "")
	getM   = CmdGet.Flag.Bool("m", false, "")
	getT   = CmdGet.Flag.Bool("t", false, "")
	getU   upgradeFlag
	// -insecure is get.Insecure
	// -v is cfg.BuildV
)

// upgradeFlag is a custom flag.Value for -u.
type upgradeFlag string

func (*upgradeFlag) IsBoolFlag() bool { return true } // allow -u

func (v *upgradeFlag) Set(s string) error {
	if s == "false" {
		s = ""
	}
	*v = upgradeFlag(s)
	return nil
}

func (v *upgradeFlag) String() string { return "" }

func init() {
	work.AddBuildFlags(CmdGet)
	CmdGet.Run = runGet // break init loop
	CmdGet.Flag.BoolVar(&get.Insecure, "insecure", get.Insecure, "")
	CmdGet.Flag.Var(&getU, "u", "")
}

type Pkg struct {
	Arg  string
	Path string
	Vers string
}

func runGet(cmd *base.Command, args []string) {
	switch getU {
	case "", "patch", "true":
		// ok
	default:
		base.Fatalf("go get: unknown upgrade flag -u=%s", getU)
	}
	if *getF {
		fmt.Fprintf(os.Stderr, "go get: -f flag is a no-op when using modules\n")
	}
	if *getFix {
		fmt.Fprintf(os.Stderr, "go get: -fix flag is a no-op when using modules\n")
	}
	if *getT {
		fmt.Fprintf(os.Stderr, "go get: -t flag is a no-op when using modules\n")
	}

	if cfg.BuildGetmode == "vendor" {
		base.Fatalf("go get: disabled by -getmode=vendor")
	}

	modload.LoadBuildList()

	// A task holds the state for processing a single get argument (path@vers).
	type task struct {
		arg             string           // original argument
		path            string           // package path part of arg
		forceModulePath bool             // path must be interpreted as a module path
		vers            string           // version part of arg
		m               module.Version   // module version indicated by argument
		req             []module.Version // m's requirement list (not upgraded)
	}

	// Build task and install lists.
	// The command-line arguments are of the form path@version
	// or simply path, with implicit @latest. path@none is "downgrade away".
	// At the end of the loop, we've resolved the list of arguments into
	// a list of tasks (a path@vers that needs further processing)
	// and a list of install targets (for the "go install" at the end).
	var tasks []*task
	var install []string
	for _, arg := range search.CleanImportPaths(args) {
		// Argument is module query path@vers, or else path with implicit @latest.
		path := arg
		vers := ""
		if i := strings.Index(arg, "@"); i >= 0 {
			path, vers = arg[:i], arg[i+1:]
		}
		if strings.Contains(vers, "@") || arg != path && vers == "" {
			base.Errorf("go get %s: invalid module version syntax", arg)
			continue
		}
		if vers != "none" {
			install = append(install, path)
		}

		// Deciding which module to upgrade/downgrade for a particular argument is difficult.
		// Patterns only make it more difficult.
		// We impose restrictions to avoid needing to interlace pattern expansion,
		// like in in modload.ImportPaths.
		// Specifically, these patterns are supported:
		//
		//	- Relative paths like ../../foo or ../../foo... are restricted to matching directories
		//	  in the current module and therefore map to the current module.
		//	  It's possible that the pattern matches no packages, but we will still treat it
		//	  as mapping to the current module.
		//	  TODO: In followup, could just expand the full list and remove the discrepancy.
		//	- The pattern "all" has its usual package meaning and maps to the list of modules
		//	  from which the matched packages are drawn. This is potentially a subset of the
		//	  module pattern "all". If module A requires B requires C but A does not import
		//	  the parts of B that import C, the packages matched by "all" are only from A and B,
		//	  so only A and B end up on the tasks list.
		//	  TODO: Even in -m mode?
		//	- The patterns "std" and "cmd" expand to packages in the standard library,
		//	  which aren't upgradable, so we skip over those.
		//	  In -m mode they expand to non-module-paths, so they are disallowed.
		//	- Import path patterns like foo/bar... are matched against the module list,
		//	  assuming any package match would imply a module pattern match.
		//	  TODO: What about -m mode?
		//	- Import paths without patterns are left as is, for resolution by getQuery (eventually modload.Import).
		//
		if search.IsRelativePath(path) {
			// Check that this relative pattern only matches directories in the current module,
			// and then record the current module as the target.
			dir := path
			if i := strings.Index(path, "..."); i >= 0 {
				dir, _ = pathpkg.Split(path[:i])
			}
			abs, err := filepath.Abs(dir)
			if err != nil {
				base.Errorf("go get %s: %v", arg, err)
				continue
			}
			if !str.HasFilePathPrefix(abs, modload.ModRoot) {
				base.Errorf("go get %s: directory %s is outside module root %s", arg, abs, modload.ModRoot)
				continue
			}
			// TODO: Check if abs is inside a nested module.
			tasks = append(tasks, &task{arg: arg, path: modload.Target.Path, vers: ""})
			continue
		}
		if path == "all" {
			if path != arg {
				base.Errorf("go get %s: cannot use pattern %q with explicit version", arg, arg)
			}

			// TODO: If *getM, should this be the module pattern "all"?

			// This is the package pattern "all" not the module pattern "all":
			// enumerate all the modules actually needed by builds of the packages
			// in the main module, not incidental modules that happen to be
			// in the package graph (and therefore build list).
			// Note that LoadALL may add new modules to the build list to
			// satisfy new imports, but vers == "latest" implicitly anyway,
			// so we'll assume that's OK.
			seen := make(map[module.Version]bool)
			pkgs := modload.LoadALL()
			for _, pkg := range pkgs {
				m := modload.PackageModule(pkg)
				if m.Path != "" && !seen[m] {
					seen[m] = true
					tasks = append(tasks, &task{arg: arg, path: m.Path, vers: "latest", forceModulePath: true})
				}
			}
			continue
		}
		if search.IsMetaPackage(path) {
			// Already handled "all", so this must be "std" or "cmd",
			// which are entirely in the standard library.
			if path != arg {
				base.Errorf("go get %s: cannot use pattern %q with explicit version", arg, arg)
			}
			if *getM {
				base.Errorf("go get %s: cannot use pattern %q with -m", arg, arg)
				continue
			}
			continue
		}
		if strings.Contains(path, "...") {
			// Apply to modules in build list matched by pattern (golang.org/x/...), if any.
			match := search.MatchPattern(path)
			matched := false
			for _, m := range modload.BuildList() {
				if match(m.Path) || str.HasPathPrefix(path, m.Path) {
					tasks = append(tasks, &task{arg: arg, path: m.Path, vers: vers, forceModulePath: true})
					matched = true
				}
			}
			// If matched, we're done.
			// Otherwise assume pattern is inside a single module
			// (golang.org/x/text/unicode/...) and leave for usual lookup.
			// Unless we're using -m.
			if matched {
				continue
			}
			if *getM {
				base.Errorf("go get %s: pattern matches no modules in build list", arg)
				continue
			}
		}
		tasks = append(tasks, &task{arg: arg, path: path, vers: vers})
	}
	base.ExitIfErrors()

	// Now we've reduced the upgrade/downgrade work to a list of path@vers pairs (tasks).
	// Resolve each one and load direct requirements in parallel.
	reqs := modload.Reqs()
	var lookup par.Work
	for _, t := range tasks {
		lookup.Add(t)
	}
	lookup.Do(10, func(item interface{}) {
		t := item.(*task)
		m, err := getQuery(t.path, t.vers, t.forceModulePath)
		if err != nil {
			base.Errorf("go get %v: %v", t.arg, err)
			return
		}
		t.m = m
		if t.vers == "none" {
			// Wait for downgrade step.
			return
		}
		// If there is no -u, then we don't need to upgrade the
		// collected requirements separately from the overall
		// recalculation of the build list (modload.ReloadBuildList below),
		// so don't bother doing it now. Doing it now wouldn't be
		// any slower (because it would prime the cache for later)
		// but the larger operation below can report more errors in a single run.
		if getU != "" {
			list, err := reqs.Required(m)
			if err != nil {
				base.Errorf("go get %v: %v", t.arg, err)
				return
			}
			t.req = list
		}
	})
	base.ExitIfErrors()

	// Now we know the specific version of each path@vers along with its requirements.
	// The final build list will be the union of three build lists:
	//	1. the original build list
	//	2. the modules named on the command line
	//	3. the upgraded requirements of those modules (if upgrading)
	// Start building those lists.
	// This loop collects (2) and the not-yet-upgraded (3).
	// Also, because the list of paths might have named multiple packages in a single module
	// (or even the same package multiple times), now that we know the module for each
	// package, this loop deduplicates multiple references to a given module.
	// (If a module is mentioned multiple times, the listed target version must be the same each time.)
	var named []module.Version
	var required []module.Version
	byPath := make(map[string]*task)
	for _, t := range tasks {
		prev, ok := byPath[t.m.Path]
		if prev != nil && prev.m != t.m {
			base.Errorf("go get: conflicting versions for module %s: %s and %s", t.m.Path, prev.m.Version, t.m.Version)
			byPath[t.m.Path] = nil // sentinel to stop errors
			continue
		}
		if ok {
			continue // already added
		}
		byPath[t.m.Path] = t
		named = append(named, t.m)
		required = append(required, t.req...)
	}
	base.ExitIfErrors()

	// If the modules named on the command line have any dependencies
	// and we're supposed to upgrade dependencies,
	// chase down the full list of upgraded dependencies.
	// This turns required from a not-yet-upgraded (3) to the final (3).
	// (See list above.)
	if len(required) > 0 {
		upgraded, err := mvs.UpgradeAll(upgradeTarget, &upgrader{
			Reqs:    modload.Reqs(),
			targets: required,
			patch:   getU == "patch",
		})
		if err != nil {
			base.Fatalf("go get: %v", err)
		}
		required = upgraded[1:] // slice off upgradeTarget
	}

	// Put together the final build list as described above (1) (2) (3).
	// If we're not using -u, then len(required) == 0 and ReloadBuildList
	// chases down the dependencies of all the named module versions
	// in one operation.
	var list []module.Version
	list = append(list, modload.BuildList()...)
	list = append(list, named...)
	list = append(list, required...)
	modload.SetBuildList(list)
	modload.ReloadBuildList() // note: does not update go.mod

	// Apply any needed downgrades.
	var down []module.Version
	for _, m := range modload.BuildList() {
		t := byPath[m.Path]
		if t != nil && semver.Compare(m.Version, t.m.Version) > 0 {
			down = append(down, module.Version{Path: m.Path, Version: t.m.Version})
		}
	}
	if len(down) > 0 {
		list, err := mvs.Downgrade(modload.Target, modload.Reqs(), down...)
		if err != nil {
			base.Fatalf("go get: %v", err)
		}
		modload.SetBuildList(list)
		modload.ReloadBuildList() // note: does not update go.mod
	}

	// Everything succeeded. Update go.mod.
	modload.WriteGoMod()

	// If -m was specified, we're done after the module work. No download, no build.
	if *getM {
		return
	}

	if len(install) > 0 {
		work.BuildInit()
		var pkgs []string
		for _, p := range load.PackagesAndErrors(install) {
			if p.Error == nil || !strings.HasPrefix(p.Error.Err, "no Go files") {
				pkgs = append(pkgs, p.ImportPath)
			}
		}
		// If -d was specified, we're done after the download: no build.
		// (The load.PackagesAndErrors is what did the download
		// of the named packages and their dependencies.)
		if len(pkgs) > 0 && !*getD {
			work.InstallPackages(pkgs)
		}
	}
}

// getQuery evaluates the given package path, version pair
// to determine the underlying module version being requested.
// If forceModulePath is set, getQuery must interpret path
// as a module path.
func getQuery(path, vers string, forceModulePath bool) (module.Version, error) {
	if path == modload.Target.Path {
		if vers != "" {
			return module.Version{}, fmt.Errorf("cannot update main module to explicit version")
		}
		return modload.Target, nil
	}

	if vers == "" {
		vers = "latest"
	}

	// First choice is always to assume path is a module path.
	// If that works out, we're done.
	info, err := modload.Query(path, vers, modload.Allowed)
	if err == nil {
		return module.Version{Path: path, Version: info.Version}, nil
	}

	// Even if the query fails, if the path is (or must be) a real module, then report the query error.
	if forceModulePath || *getM || isModulePath(path) {
		return module.Version{}, err
	}

	// Otherwise, interpret the package path as an import
	// and determine what module that import would address
	// if found in the current source code.
	// Then apply the version to that module.
	m, _, err := modload.Import(path)
	if err != nil {
		return module.Version{}, err
	}
	if m.Path == "" {
		return module.Version{}, fmt.Errorf("package %q is not in a module", path)
	}
	info, err = modload.Query(m.Path, vers, modload.Allowed)
	if err != nil {
		return module.Version{}, err
	}
	return module.Version{Path: m.Path, Version: info.Version}, nil
}

// isModulePath reports whether path names an actual module,
// defined as one with an accessible latest version.
func isModulePath(path string) bool {
	_, err := modload.Query(path, "latest", modload.Allowed)
	return err == nil
}

// An upgrader adapts an underlying mvs.Reqs to apply an
// upgrade policy to a list of targets and their dependencies.
// If patch=false, the upgrader implements "get -u".
// If patch=true, the upgrader implements "get -u=patch".
type upgrader struct {
	mvs.Reqs
	targets []module.Version
	patch   bool
}

// upgradeTarget is a fake "target" requiring all the modules to be upgraded.
var upgradeTarget = module.Version{Path: "upgrade target", Version: ""}

// Required returns the requirement list for m.
// Other than the upgradeTarget, we defer to u.Reqs.
func (u *upgrader) Required(m module.Version) ([]module.Version, error) {
	if m == upgradeTarget {
		return u.targets, nil
	}
	return u.Reqs.Required(m)
}

// Upgrade returns the desired upgrade for m.
// If m is a tagged version, then Upgrade returns the latest tagged version.
// If m is a pseudo-version, then Upgrade returns the latest tagged version
// when that version has a time-stamp newer than m.
// Otherwise Upgrade returns m (preserving the pseudo-version).
// This special case prevents accidental downgrades
// when already using a pseudo-version newer than the latest tagged version.
func (u *upgrader) Upgrade(m module.Version) (module.Version, error) {
	// Note that query "latest" is not the same as
	// using repo.Latest.
	// The query only falls back to untagged versions
	// if nothing is tagged. The Latest method
	// only ever returns untagged versions,
	// which is not what we want.
	query := "latest"
	if u.patch {
		// For patch upgrade, query "v1.2".
		query = semver.MajorMinor(m.Version)
	}
	info, err := modload.Query(m.Path, query, modload.Allowed)
	if err != nil {
		// Report error but return m, to let version selection continue.
		// (Reporting the error will fail the command at the next base.ExitIfErrors.)
		// Special case: if the error is "no matching versions" then don't
		// even report the error. Because Query does not consider pseudo-versions,
		// it may happen that we have a pseudo-version but during -u=patch
		// the query v0.0 matches no versions (not even the one we're using).
		if !strings.Contains(err.Error(), "no matching versions") {
			base.Errorf("go get: upgrading %s@%s: %v", m.Path, m.Version, err)
		}
		return m, nil
	}

	// If we're on a later prerelease, keep using it,
	// even though normally an Upgrade will ignore prereleases.
	if semver.Compare(info.Version, m.Version) < 0 {
		return m, nil
	}

	// If we're on a pseudo-version chronologically after the latest tagged version, keep using it.
	// This avoids some accidental downgrades.
	if mTime, err := modfetch.PseudoVersionTime(m.Version); err == nil && info.Time.Before(mTime) {
		return m, nil
	}
	return module.Version{Path: m.Path, Version: info.Version}, nil
}
