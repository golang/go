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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

var CmdGet = &base.Command{
	// Note: -d -m -u are listed explicitly because they are the most common get flags.
	// Do not send CLs removing them because they're covered by [get flags].
	UsageLine: "go get [-d] [-m] [-t] [-u] [-v] [-insecure] [build flags] [packages]",
	Short:     "add dependencies to current module and install them",
	Long: `
Get resolves and adds dependencies to the current development module
and then builds and installs them.

The first step is to resolve which dependencies to add.

For each named package or package pattern, get must decide which version of
the corresponding module to use. By default, get chooses the latest tagged
release version, such as v0.4.5 or v1.2.3. If there are no tagged release
versions, get chooses the latest tagged pre-release version, such as
v0.0.1-pre1. If there are no tagged versions at all, get chooses the latest
known commit.

This default version selection can be overridden by adding an @version
suffix to the package argument, as in 'go get golang.org/x/text@v0.3.0'.
For modules stored in source control repositories, the version suffix can
also be a commit hash, branch identifier, or other syntax known to the
source control system, as in 'go get golang.org/x/text@master'.

If a module under consideration is already a dependency of the current
development module, then get will update the required version.
Specifying a version earlier than the current required version is valid and
downgrades the dependency. The version suffix @none indicates that the
dependency should be removed entirely, downgrading or removing modules
depending on it as needed.

The version suffix @latest explicitly requests the latest minor release of the
given path.

The suffix @patch requests the latest patch release: if the path is already in
the build list, the selected version will have the same minor version.
If the path is not already in the build list, @patch is equivalent to @latest.

Although get defaults to using the latest version of the module containing
a named package, it does not use the latest version of that module's
dependencies. Instead it prefers to use the specific dependency versions
requested by that module. For example, if the latest A requires module
B v1.2.3, while B v1.2.4 and v1.3.1 are also available, then 'go get A'
will use the latest A but then use B v1.2.3, as requested by A. (If there
are competing requirements for a particular module, then 'go get' resolves
those requirements by taking the maximum requested version.)

The -t flag instructs get to consider modules needed to build tests of
packages specified on the command line.

The -u flag instructs get to update dependencies to use newer minor or
patch releases when available. Continuing the previous example,
'go get -u A' will use the latest A with B v1.3.1 (not B v1.2.3).

The -u=patch flag (not -u patch) also instructs get to update dependencies,
but changes the default to select patch releases.
Continuing the previous example,
'go get -u=patch A@latest' will use the latest A with B v1.2.4 (not B v1.2.3),
while 'go get -u=patch A' will use a patch release of A instead.

When the -t and -u flags are used together, get will update
test dependencies as well.

In general, adding a new dependency may require upgrading
existing dependencies to keep a working build, and 'go get' does
this automatically. Similarly, downgrading one dependency may
require downgrading other dependencies, and 'go get' does
this automatically as well.

The -m flag instructs get to stop here, after resolving, upgrading,
and downgrading modules and updating go.mod. When using -m,
each specified package path must be a module path as well,
not the import path of a package below the module root.

When the -m and -u flags are used together, 'go get' will upgrade
modules that provide packages depended on by the modules named on
the command line. For example, 'go get -u -m A' will upgrade A and
any module providing packages imported by packages in A.
'go get -u -m' will upgrade modules that provided packages needed
by the main module.

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

With no package arguments, 'go get' applies to Go package in the
current directory, if any. In particular, 'go get -u' and
'go get -u=patch' update all the dependencies of that package.
With no package arguments and also without -u, 'go get' is not much more
than 'go install', and 'go get -d' not much more than 'go list'.

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
	if s == "true" {
		s = "latest"
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

// A getArg holds a parsed positional argument for go get (path@vers).
type getArg struct {
	// raw is the original argument, to be printed in error messages.
	raw string

	// path is the part of the argument before "@" (or the whole argument
	// if there is no "@"). path specifies the modules or packages to get.
	path string

	// vers is the part of the argument after "@" (or "" if there is no "@").
	// vers specifies the module version to get.
	vers string
}

// querySpec describes a query for a specific module. path may be a
// module path, package path, or package pattern. vers is a version
// query string from a command line argument.
type querySpec struct {
	// path is a module path, package path, or package pattern that
	// specifies which module to query.
	path string

	// vers specifies what version of the module to get.
	vers string

	// forceModulePath is true if path should be interpreted as a module path
	// even if -m is not specified.
	forceModulePath bool

	// prevM is the previous version of the module. prevM is needed
	// if vers is "patch", and the module was previously in the build list.
	prevM module.Version
}

// query holds the state for a query made for a specific module.
// After a query is performed, we know the actual module path and
// version and whether any packages were matched by the query path.
type query struct {
	querySpec

	// arg is the command line argument that matched the specified module.
	arg string

	// m is the module path and version found by the query.
	m module.Version
}

func runGet(cmd *base.Command, args []string) {
	// -mod=readonly has no effect on "go get".
	if cfg.BuildMod == "readonly" {
		cfg.BuildMod = ""
	}

	switch getU {
	case "", "latest", "patch":
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
	modload.LoadTests = *getT

	if cfg.BuildMod == "vendor" {
		base.Fatalf("go get: disabled by -mod=%s", cfg.BuildMod)
	}

	modload.LoadBuildList()

	// Do not allow any updating of go.mod until we've applied
	// all the requested changes and checked that the result matches
	// what was requested.
	modload.DisallowWriteGoMod()

	// Parse command-line arguments and report errors. The command-line
	// arguments are of the form path@version or simply path, with implicit
	// @latest. path@none is "downgrade away".
	var gets []getArg
	var queries []*query
	for _, arg := range search.CleanPatterns(args) {
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

		// If the user runs 'go get -u=patch some/module', update some/module to a
		// patch release, not a minor version.
		if vers == "" && getU != "" {
			vers = string(getU)
		}

		gets = append(gets, getArg{raw: arg, path: path, vers: vers})

		// Determine the modules that path refers to, and create queries
		// to lookup modules at target versions before loading packages.
		// This is an imprecise process, but it helps reduce unnecessary
		// queries and package loading. It's also necessary for handling
		// patterns like golang.org/x/tools/..., which can't be expanded
		// during package loading until they're in the build list.
		switch {
		case search.IsRelativePath(path):
			// Relative paths like ../../foo or ../../foo... are restricted to
			// matching packages in the main module. If the path is explicit and
			// contains no wildcards (...), check that it is a package in
			// the main module. If the path contains wildcards but matches no
			// packages, we'll warn after package loading.
			if len(args) > 0 && *getM {
				base.Errorf("go get %s: -m requires a module path, but a relative path must be a package in the main module", arg)
				continue
			}

			if !*getM && !strings.Contains(path, "...") {
				pkgPath := modload.DirImportPath(filepath.FromSlash(path))
				if pkgs := modload.TargetPackages(pkgPath); len(pkgs) == 0 {
					abs, err := filepath.Abs(path)
					if err != nil {
						abs = path
					}
					base.Errorf("go get %s: path %s is not a package in module rooted at %s", arg, abs, modload.ModRoot())
					continue
				}
			}

			if path != arg {
				base.Errorf("go get %s: can't request explicit version of path in main module", arg)
				continue
			}

		case strings.Contains(path, "..."):
			// If we're using -m, look up modules in the build list that match
			// the pattern. Report an error if no modules match.
			if *getM {
				match := search.MatchPattern(path)
				matched := false
				for _, m := range modload.BuildList() {
					if match(m.Path) || str.HasPathPrefix(path, m.Path) {
						queries = append(queries, &query{querySpec: querySpec{path: m.Path, vers: vers, prevM: m, forceModulePath: true}, arg: arg})
						matched = true
					}
				}
				if !matched {
					base.Errorf("go get %s: pattern matches no modules in build list", arg)
					continue
				}
				break
			}

			// If we're not using -m, wait until we load packages to look up modules.
			// We don't know yet whether any modules in the build list provide
			// packages matching the pattern. For example, suppose
			// golang.org/x/tools and golang.org/x/tools/playground are separate
			// modules, and only golang.org/x/tools is in the build list. If the
			// user runs 'go get golang.org/x/tools/playground/...', we should
			// add a requirement for golang.org/x/tools/playground. We should not
			// upgrade golang.org/x/tools.

		case path == "all":
			// This is the package pattern "all" not the module pattern "all",
			// even if *getM. We won't create any queries yet, since we're going to
			// need to load packages anyway.

		case search.IsMetaPackage(path):
			base.Errorf("go get %s: explicit requirement on standard-library module %s not allowed", path, path)
			continue

		default:
			// The argument is a package path or module path or both.
			q := &query{querySpec: querySpec{path: path, vers: vers}, arg: arg}
			if vers == "patch" {
				if *getM {
					for _, m := range modload.BuildList() {
						if m.Path == path {
							q.prevM = m
							break
						}
					}
					queries = append(queries, q)
				} else {
					// We need to know the module containing path before asking for
					// a specific version. Wait until we load packages later.
				}
			} else {
				// The requested version of path doesn't depend on the existing version,
				// so don't bother resolving it.
				queries = append(queries, q)
			}
		}
	}
	base.ExitIfErrors()

	// Query modules referenced by command line arguments at requested versions,
	// and add them to the build list. We need to do this before loading packages
	// since patterns that refer to packages in unknown modules can't be
	// expanded. This also avoids looking up new modules while loading packages,
	// only to downgrade later.
	queryCache := make(map[querySpec]*query)
	byPath := runQueries(queryCache, queries, nil)

	// Add queried modules to the build list. This prevents some additional
	// lookups for modules at "latest" when we load packages later.
	buildList, err := mvs.UpgradeAll(modload.Target, newUpgrader(byPath, nil))
	if err != nil {
		base.Fatalf("go get: %v", err)
	}
	modload.SetBuildList(buildList)
	base.ExitIfErrors()
	prevBuildList := buildList

	// Build a set of module paths that we don't plan to load packages from.
	// This includes explicitly requested modules that don't have a root package
	// and modules with a target version of "none".
	var wg sync.WaitGroup
	modOnly := make(map[string]*query)
	for _, q := range queries {
		if q.m.Version == "none" {
			modOnly[q.m.Path] = q
			continue
		}
		if !*getM && q.path == q.m.Path {
			wg.Add(1)
			go func(q *query) {
				if hasPkg, err := modload.ModuleHasRootPackage(q.m); err != nil {
					base.Errorf("go get: %v", err)
				} else if !hasPkg {
					modOnly[q.m.Path] = q
				}
				wg.Done()
			}(q)
		}
	}
	wg.Wait()
	base.ExitIfErrors()

	// Build a list of arguments that may refer to packages.
	var pkgPatterns []string
	var pkgGets []getArg
	for _, arg := range gets {
		if modOnly[arg.path] == nil && arg.vers != "none" {
			pkgPatterns = append(pkgPatterns, arg.path)
			pkgGets = append(pkgGets, arg)
		}
	}

	// Load packages and upgrade the modules that provide them. We do this until
	// we reach a fixed point, since modules providing packages may change as we
	// change versions. This must terminate because the module graph is finite,
	// and the load and upgrade operations may only add and upgrade modules
	// in the build list.
	var matches []*search.Match
	var install []string
	for {
		var seenPkgs map[string]bool
		seenQuery := make(map[querySpec]bool)
		var queries []*query
		addQuery := func(q *query) {
			if !seenQuery[q.querySpec] {
				seenQuery[q.querySpec] = true
				queries = append(queries, q)
			}
		}

		if len(pkgPatterns) > 0 {
			// Don't load packages if pkgPatterns is empty. Both
			// modload.ImportPathsQuiet and ModulePackages convert an empty list
			// of patterns to []string{"."}, which is not what we want.
			if *getM {
				matches = modload.ModulePackages(pkgPatterns)
			} else {
				matches = modload.ImportPathsQuiet(pkgPatterns)
			}
			seenPkgs = make(map[string]bool)
			install = make([]string, 0, len(pkgPatterns))
			for i, match := range matches {
				arg := pkgGets[i]

				if !*getM && len(match.Pkgs) == 0 {
					// If the pattern did not match any packages, look up a new module.
					// If the pattern doesn't match anything on the last iteration,
					// we'll print a warning after the outer loop.
					if !search.IsRelativePath(arg.path) && !match.Literal && arg.path != "all" {
						addQuery(&query{querySpec: querySpec{path: arg.path, vers: arg.vers}, arg: arg.raw})
					}
					continue
				}

				install = append(install, arg.path)
				allStd := true
				for _, pkg := range match.Pkgs {
					if !seenPkgs[pkg] {
						seenPkgs[pkg] = true
						if _, _, err := modload.Lookup("", false, pkg); err != nil {
							allStd = false
							base.Errorf("go get %s: %v", arg.raw, err)
							continue
						}
					}
					m := modload.PackageModule(pkg)
					if m.Path == "" {
						// pkg is in the standard library.
						continue
					}
					allStd = false
					addQuery(&query{querySpec: querySpec{path: m.Path, vers: arg.vers, forceModulePath: true, prevM: m}, arg: arg.raw})
				}
				if allStd {
					if *getM {
						base.Errorf("go get %s: cannot use pattern %q with -m", arg.raw, arg.raw)
					} else if arg.path != arg.raw {
						base.Errorf("go get %s: cannot use pattern %q with explicit version", arg.raw, arg.raw)
					}
				}
			}
		}
		base.ExitIfErrors()

		// Query target versions for modules providing packages matched by
		// command line arguments.
		byPath = runQueries(queryCache, queries, modOnly)

		// Handle upgrades. This is needed for arguments that didn't match
		// modules or matched different modules from a previous iteration. It
		// also upgrades modules providing package dependencies if -u is set.
		buildList, err := mvs.UpgradeAll(modload.Target, newUpgrader(byPath, seenPkgs))
		if err != nil {
			base.Fatalf("go get: %v", err)
		}
		modload.SetBuildList(buildList)
		base.ExitIfErrors()

		// Stop if no changes have been made to the build list.
		buildList = modload.BuildList()
		eq := len(buildList) == len(prevBuildList)
		for i := 0; eq && i < len(buildList); i++ {
			eq = buildList[i] == prevBuildList[i]
		}
		if eq {
			break
		}
		prevBuildList = buildList
	}
	if !*getM {
		search.WarnUnmatched(matches) // don't warn on every iteration
	}

	// Handle downgrades.
	var down []module.Version
	for _, m := range modload.BuildList() {
		q := byPath[m.Path]
		if q != nil && semver.Compare(m.Version, q.m.Version) > 0 {
			down = append(down, module.Version{Path: m.Path, Version: q.m.Version})
		}
	}
	if len(down) > 0 {
		buildList, err := mvs.Downgrade(modload.Target, modload.Reqs(), down...)
		if err != nil {
			base.Fatalf("go: %v", err)
		}
		modload.SetBuildList(buildList)
		modload.ReloadBuildList() // note: does not update go.mod
		base.ExitIfErrors()
	}

	// Scan for any upgrades lost by the downgrades.
	var lostUpgrades []*query
	var versionByPath map[string]string
	if len(down) > 0 {
		versionByPath = make(map[string]string)
		for _, m := range modload.BuildList() {
			versionByPath[m.Path] = m.Version
		}
		for _, q := range byPath {
			if v, ok := versionByPath[q.m.Path]; q.m.Version != "none" && (!ok || semver.Compare(v, q.m.Version) != 0) {
				lostUpgrades = append(lostUpgrades, q)
			}
		}
		sort.Slice(lostUpgrades, func(i, j int) bool {
			return lostUpgrades[i].m.Path < lostUpgrades[j].m.Path
		})
	}
	if len(lostUpgrades) > 0 {
		desc := func(m module.Version) string {
			s := m.Path + "@" + m.Version
			t := byPath[m.Path]
			if t != nil && t.arg != s {
				s += " from " + t.arg
			}
			return s
		}
		downByPath := make(map[string]module.Version)
		for _, d := range down {
			downByPath[d.Path] = d
		}

		var buf strings.Builder
		fmt.Fprintf(&buf, "go get: inconsistent versions:")
		reqs := modload.Reqs()
		for _, q := range lostUpgrades {
			// We lost q because its build list requires a newer version of something in down.
			// Figure out exactly what.
			// Repeatedly constructing the build list is inefficient
			// if there are MANY command-line arguments,
			// but at least all the necessary requirement lists are cached at this point.
			list, err := buildListForLostUpgrade(q.m, reqs)
			if err != nil {
				base.Fatalf("go: %v", err)
			}

			fmt.Fprintf(&buf, "\n\t%s", desc(q.m))
			sep := " requires"
			for _, m := range list {
				if down, ok := downByPath[m.Path]; ok && semver.Compare(down.Version, m.Version) < 0 {
					fmt.Fprintf(&buf, "%s %s@%s (not %s)", sep, m.Path, m.Version, desc(down))
					sep = ","
				}
			}
			if sep != "," {
				// We have no idea why this happened.
				// At least report the problem.
				if v := versionByPath[q.m.Path]; v == "" {
					fmt.Fprintf(&buf, " removed unexpectedly")
				} else {
					fmt.Fprintf(&buf, " ended up at %s unexpectedly", v)
				}
				fmt.Fprintf(&buf, " (please report at golang.org/issue/new)")
			}
		}
		base.Fatalf("%v", buf.String())
	}

	// Everything succeeded. Update go.mod.
	modload.AllowWriteGoMod()
	modload.WriteGoMod()

	// If -m or -d was specified, we're done after the module work. We've
	// already downloaded modules by loading packages above. If neither flag
	// we specified, we need build and install the packages.
	// Note that 'go get -u' without any arguments results in len(install) == 1:
	// search.CleanImportPaths returns "." for empty args.
	if *getM || *getD || len(install) == 0 {
		return
	}
	work.BuildInit()
	pkgs := load.PackagesForBuild(install)
	work.InstallPackages(install, pkgs)
}

// runQueries looks up modules at target versions in parallel. Results will be
// cached. If the same module is referenced by multiple queries at different
// versions (including earlier queries in the modOnly map), an error will be
// reported. A map from module paths to queries is returned, which includes
// queries and modOnly.
func runQueries(cache map[querySpec]*query, queries []*query, modOnly map[string]*query) map[string]*query {
	var lookup par.Work
	for _, q := range queries {
		if cached := cache[q.querySpec]; cached != nil {
			*q = *cached
		} else {
			cache[q.querySpec] = q
			lookup.Add(q)
		}
	}

	lookup.Do(10, func(item interface{}) {
		q := item.(*query)
		if q.vers == "none" {
			// Wait for downgrade step.
			q.m = module.Version{Path: q.path, Version: "none"}
			return
		}
		m, err := getQuery(q.path, q.vers, q.prevM, q.forceModulePath)
		if err != nil {
			base.Errorf("go get %s: %v", q.arg, err)
		}
		q.m = m
	})
	base.ExitIfErrors()

	byPath := make(map[string]*query)
	check := func(q *query) {
		if prev, ok := byPath[q.m.Path]; prev != nil && prev.m != q.m {
			base.Errorf("go get: conflicting versions for module %s: %s and %s", q.m.Path, prev.m.Version, q.m.Version)
			byPath[q.m.Path] = nil // sentinel to stop errors
			return
		} else if !ok {
			byPath[q.m.Path] = q
		}
	}
	for _, q := range queries {
		check(q)
	}
	for _, q := range modOnly {
		check(q)
	}
	base.ExitIfErrors()

	return byPath
}

// getQuery evaluates the given package path, version pair
// to determine the underlying module version being requested.
// If forceModulePath is set, getQuery must interpret path
// as a module path.
func getQuery(path, vers string, prevM module.Version, forceModulePath bool) (module.Version, error) {
	switch vers {
	case "":
		vers = "latest"
	case "patch":
		if prevM.Version == "" {
			vers = "latest"
		} else {
			vers = semver.MajorMinor(prevM.Version)
		}
	}

	if forceModulePath || *getM || !strings.Contains(path, "...") {
		if path == modload.Target.Path {
			if vers != "latest" {
				return module.Version{}, fmt.Errorf("can't get a specific version of the main module")
			}
		}

		// If the path doesn't contain a wildcard, try interpreting it as a module path.
		info, err := modload.Query(path, vers, modload.Allowed)
		if err == nil {
			return module.Version{Path: path, Version: info.Version}, nil
		}

		// If the query fails, and the path must be a real module, report the query error.
		if forceModulePath || *getM {
			return module.Version{}, err
		}
	}

	// Otherwise, try a package path or pattern.
	results, err := modload.QueryPattern(path, vers, modload.Allowed)
	if err != nil {
		return module.Version{}, err
	}
	return results[0].Mod, nil
}

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
	// Note that query "latest" is not the same as
	// using repo.Latest.
	// The query only falls back to untagged versions
	// if nothing is tagged. The Latest method
	// only ever returns untagged versions,
	// which is not what we want.
	query := "latest"
	if getU == "patch" {
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
		var noMatch *modload.NoMatchingVersionError
		if !errors.As(err, &noMatch) {
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
