// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package modget implements the module-aware ``go get'' command.
package modget

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/get"
	"cmd/go/internal/imports"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/mvs"
	"cmd/go/internal/search"
	"cmd/go/internal/work"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

var CmdGet = &base.Command{
	// Note: -d -u are listed explicitly because they are the most common get flags.
	// Do not send CLs removing them because they're covered by [get flags].
	UsageLine: "go get [-d] [-t] [-u] [-v] [-insecure] [build flags] [packages]",
	Short:     "add dependencies to current module and install them",
	Long: `
Get resolves and adds dependencies to the current development module
and then builds and installs them.

The first step is to resolve which dependencies to add.

For each named package or package pattern, get must decide which version of
the corresponding module to use. By default, get looks up the latest tagged
release version, such as v0.4.5 or v1.2.3. If there are no tagged release
versions, get looks up the latest tagged pre-release version, such as
v0.0.1-pre1. If there are no tagged versions at all, get looks up the latest
known commit. If the module is not already required at a later version
(for example, a pre-release newer than the latest release), get will use
the version it looked up. Otherwise, get will use the currently
required version.

This default version selection can be overridden by adding an @version
suffix to the package argument, as in 'go get golang.org/x/text@v0.3.0'.
The version may be a prefix: @v1 denotes the latest available version starting
with v1. See 'go help modules' under the heading 'Module queries' for the
full query syntax.

For modules stored in source control repositories, the version suffix can
also be a commit hash, branch identifier, or other syntax known to the
source control system, as in 'go get golang.org/x/text@master'. Note that
branches with names that overlap with other module query syntax cannot be
selected explicitly. For example, the suffix @v2 means the latest version
starting with v2, not the branch named v2.

If a module under consideration is already a dependency of the current
development module, then get will update the required version.
Specifying a version earlier than the current required version is valid and
downgrades the dependency. The version suffix @none indicates that the
dependency should be removed entirely, downgrading or removing modules
depending on it as needed.

The version suffix @latest explicitly requests the latest minor release of the
module named by the given path. The suffix @upgrade is like @latest but
will not downgrade a module if it is already required at a revision or
pre-release version newer than the latest released version. The suffix
@patch requests the latest patch release: the latest released version
with the same major and minor version numbers as the currently required
version. Like @upgrade, @patch will not downgrade a module already required
at a newer version. If the path is not already required, @upgrade and @patch
are equivalent to @latest.

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

The -u flag instructs get to update modules providing dependencies
of packages named on the command line to use newer minor or patch
releases when available. Continuing the previous example, 'go get -u A'
will use the latest A with B v1.3.1 (not B v1.2.3). If B requires module C,
but C does not provide any packages needed to build packages in A
(not including tests), then C will not be updated.

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

The -insecure flag permits fetching from repositories and resolving
custom domains using insecure schemes such as HTTP. Use with caution. The
GOINSECURE environment variable is usually a better alternative, since it
provides control over which modules may be retrieved using an insecure scheme.
See 'go help environment' for details.

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
		s = "upgrade"
	}
	*v = upgradeFlag(s)
	return nil
}

func (v *upgradeFlag) String() string { return "" }

func init() {
	work.AddBuildFlags(CmdGet, work.OmitModFlag)
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

	// vers is the part of the argument after "@" or an implied
	// "upgrade" or "patch" if there is no "@". vers specifies the
	// module version to get.
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

	// forceModulePath is true if path should be interpreted as a module path.
	// If forceModulePath is true, prevM must be set.
	forceModulePath bool

	// prevM is the previous version of the module. prevM is needed
	// to determine the minor version number if vers is "patch". It's also
	// used to avoid downgrades from prerelease versions newer than
	// "latest" and "patch". If prevM is set, forceModulePath must be true.
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

func runGet(ctx context.Context, cmd *base.Command, args []string) {
	switch getU {
	case "", "upgrade", "patch":
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
	if *getM {
		base.Fatalf("go get: -m flag is no longer supported; consider -d to skip building packages")
	}
	modload.LoadTests = *getT

	buildList := modload.LoadAllModules(ctx)
	buildList = buildList[:len(buildList):len(buildList)] // copy on append
	versionByPath := make(map[string]string)
	for _, m := range buildList {
		versionByPath[m.Path] = m.Version
	}

	// Do not allow any updating of go.mod until we've applied
	// all the requested changes and checked that the result matches
	// what was requested.
	modload.DisallowWriteGoMod()

	// Allow looking up modules for import paths when outside of a module.
	// 'go get' is expected to do this, unlike other commands.
	modload.AllowMissingModuleImports()

	// Parse command-line arguments and report errors. The command-line
	// arguments are of the form path@version or simply path, with implicit
	// @upgrade. path@none is "downgrade away".
	var gets []getArg
	var queries []*query
	for _, arg := range search.CleanPatterns(args) {
		// Argument is path or path@vers.
		path := arg
		vers := ""
		if i := strings.Index(arg, "@"); i >= 0 {
			path, vers = arg[:i], arg[i+1:]
		}
		if strings.Contains(vers, "@") || arg != path && vers == "" {
			base.Errorf("go get %s: invalid module version syntax", arg)
			continue
		}

		// Guard against 'go get x.go', a common mistake.
		// Note that package and module paths may end with '.go', so only print an error
		// if the argument has no version and either has no slash or refers to an existing file.
		if strings.HasSuffix(arg, ".go") && vers == "" {
			if !strings.Contains(arg, "/") {
				base.Errorf("go get %s: arguments must be package or module paths", arg)
				continue
			}
			if fi, err := os.Stat(arg); err == nil && !fi.IsDir() {
				base.Errorf("go get: %s exists as a file, but 'go get' requires package arguments", arg)
				continue
			}
		}

		// If no version suffix is specified, assume @upgrade.
		// If -u=patch was specified, assume @patch instead.
		if vers == "" {
			if getU != "" {
				vers = string(getU)
			} else {
				vers = "upgrade"
			}
		}

		gets = append(gets, getArg{raw: arg, path: path, vers: vers})

		// Determine the modules that path refers to, and create queries
		// to lookup modules at target versions before loading packages.
		// This is an imprecise process, but it helps reduce unnecessary
		// queries and package loading. It's also necessary for handling
		// patterns like golang.org/x/tools/..., which can't be expanded
		// during package loading until they're in the build list.
		switch {
		case filepath.IsAbs(path) || search.IsRelativePath(path):
			// Absolute paths like C:\foo and relative paths like ../foo...
			// are restricted to matching packages in the main module. If the path
			// is explicit and contains no wildcards (...), check that it is a
			// package in the main module. If the path contains wildcards but
			// matches no packages, we'll warn after package loading.
			if !strings.Contains(path, "...") {
				m := search.NewMatch(path)
				if pkgPath := modload.DirImportPath(path); pkgPath != "." {
					m = modload.TargetPackages(ctx, pkgPath)
				}
				if len(m.Pkgs) == 0 {
					for _, err := range m.Errs {
						base.Errorf("go get %s: %v", arg, err)
					}

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
			// Wait until we load packages to look up modules.
			// We don't know yet whether any modules in the build list provide
			// packages matching the pattern. For example, suppose
			// golang.org/x/tools and golang.org/x/tools/playground are separate
			// modules, and only golang.org/x/tools is in the build list. If the
			// user runs 'go get golang.org/x/tools/playground/...', we should
			// add a requirement for golang.org/x/tools/playground. We should not
			// upgrade golang.org/x/tools.

		case path == "all":
			// If there is no main module, "all" is not meaningful.
			if !modload.HasModRoot() {
				base.Errorf(`go get %s: cannot match "all": working directory is not part of a module`, arg)
			}
			// Don't query modules until we load packages. We'll automatically
			// look up any missing modules.

		case search.IsMetaPackage(path):
			base.Errorf("go get %s: explicit requirement on standard-library module %s not allowed", path, path)
			continue

		default:
			// The argument is a package or module path.
			if modload.HasModRoot() {
				if m := modload.TargetPackages(ctx, path); len(m.Pkgs) != 0 {
					// The path is in the main module. Nothing to query.
					if vers != "upgrade" && vers != "patch" {
						base.Errorf("go get %s: can't request explicit version of path in main module", arg)
					}
					continue
				}
			}

			first := path
			if i := strings.IndexByte(first, '/'); i >= 0 {
				first = path
			}
			if !strings.Contains(first, ".") {
				// The path doesn't have a dot in the first component and cannot be
				// queried as a module. It may be a package in the standard library,
				// which is fine, so don't report an error unless we encounter
				// a problem loading packages below.
				continue
			}

			// If we're querying "upgrade" or "patch", we need to know the current
			// version of the module. For "upgrade", we want to avoid accidentally
			// downgrading from a newer prerelease. For "patch", we need to query
			// the correct minor version.
			// Here, we check if "path" is the name of a module in the build list
			// (other than the main module) and set prevM if so. If "path" isn't
			// a module in the build list, the current version doesn't matter
			// since it's either an unknown module or a package within a module
			// that we'll discover later.
			q := &query{querySpec: querySpec{path: path, vers: vers}, arg: arg}
			if v, ok := versionByPath[path]; ok && path != modload.Target.Path {
				q.prevM = module.Version{Path: path, Version: v}
				q.forceModulePath = true
			}
			queries = append(queries, q)
		}
	}
	base.ExitIfErrors()

	// Query modules referenced by command line arguments at requested versions.
	// We need to do this before loading packages since patterns that refer to
	// packages in unknown modules can't be expanded. This also avoids looking
	// up new modules while loading packages, only to downgrade later.
	queryCache := make(map[querySpec]*query)
	byPath := runQueries(ctx, queryCache, queries, nil)

	// Add missing modules to the build list.
	// We call SetBuildList here and elsewhere, since newUpgrader,
	// ImportPathsQuiet, and other functions read the global build list.
	for _, q := range queries {
		if _, ok := versionByPath[q.m.Path]; !ok && q.m.Version != "none" {
			buildList = append(buildList, q.m)
		}
	}
	versionByPath = nil // out of date now; rebuilt later when needed
	modload.SetBuildList(buildList)

	// Upgrade modules specifically named on the command line. This is our only
	// chance to upgrade modules without root packages (modOnly below).
	// This also skips loading packages at an old version, only to upgrade
	// and reload at a new version.
	upgrade := make(map[string]*query)
	for path, q := range byPath {
		if q.path == q.m.Path && q.m.Version != "none" {
			upgrade[path] = q
		}
	}
	buildList, err := mvs.UpgradeAll(modload.Target, newUpgrader(upgrade, nil))
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
	var modOnlyMu sync.Mutex
	modOnly := make(map[string]*query)
	for _, q := range queries {
		if q.m.Version == "none" {
			modOnlyMu.Lock()
			modOnly[q.m.Path] = q
			modOnlyMu.Unlock()
			continue
		}
		if q.path == q.m.Path {
			wg.Add(1)
			go func(q *query) {
				if hasPkg, err := modload.ModuleHasRootPackage(ctx, q.m); err != nil {
					base.Errorf("go get: %v", err)
				} else if !hasPkg {
					modOnlyMu.Lock()
					modOnly[q.m.Path] = q
					modOnlyMu.Unlock()
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
			matches = modload.ImportPathsQuiet(ctx, pkgPatterns, imports.AnyTags())
			seenPkgs = make(map[string]bool)
			for i, match := range matches {
				arg := pkgGets[i]

				if len(match.Pkgs) == 0 {
					// If the pattern did not match any packages, look up a new module.
					// If the pattern doesn't match anything on the last iteration,
					// we'll print a warning after the outer loop.
					if !match.IsLocal() && !match.IsLiteral() && arg.path != "all" {
						addQuery(&query{querySpec: querySpec{path: arg.path, vers: arg.vers}, arg: arg.raw})
					} else {
						for _, err := range match.Errs {
							base.Errorf("go get: %v", err)
						}
					}
					continue
				}

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
					if m.Path == modload.Target.Path {
						// pkg is in the main module.
						continue
					}
					addQuery(&query{querySpec: querySpec{path: m.Path, vers: arg.vers, forceModulePath: true, prevM: m}, arg: arg.raw})
				}
				if allStd && arg.path != arg.raw {
					base.Errorf("go get %s: cannot use pattern %q with explicit version", arg.raw, arg.raw)
				}
			}
		}
		base.ExitIfErrors()

		// Query target versions for modules providing packages matched by
		// command line arguments.
		byPath = runQueries(ctx, queryCache, queries, modOnly)

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
		buildList = modload.LoadedModules()
		eq := len(buildList) == len(prevBuildList)
		for i := 0; eq && i < len(buildList); i++ {
			eq = buildList[i] == prevBuildList[i]
		}
		if eq {
			break
		}
		prevBuildList = buildList
	}
	if !*getD {
		// Only print warnings after the last iteration,
		// and only if we aren't going to build.
		search.WarnUnmatched(matches)
	}

	// Handle downgrades.
	var down []module.Version
	for _, m := range modload.LoadedModules() {
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

		// TODO(bcmills) What should happen here under lazy loading?
		// Downgrading may intentionally violate the lazy-loading invariants.

		modload.SetBuildList(buildList)
		modload.ReloadBuildList() // note: does not update go.mod
		base.ExitIfErrors()
	}

	// Scan for any upgrades lost by the downgrades.
	var lostUpgrades []*query
	if len(down) > 0 {
		versionByPath = make(map[string]string)
		for _, m := range modload.LoadedModules() {
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
	modload.DisallowWriteGoMod()

	// Report warnings if any retracted versions are in the build list.
	// This must be done after writing go.mod to avoid spurious '// indirect'
	// comments. These functions read and write global state.
	// TODO(golang.org/issue/40775): ListModules resets modload.loader, which
	// contains information about direct dependencies that WriteGoMod uses.
	// Refactor to avoid these kinds of global side effects.
	reportRetractions(ctx)

	// If -d was specified, we're done after the module work.
	// We've already downloaded modules by loading packages above.
	// Otherwise, we need to build and install the packages matched by
	// command line arguments. This may be a different set of packages,
	// since we only build packages for the target platform.
	// Note that 'go get -u' without arguments is equivalent to
	// 'go get -u .', so we'll typically build the package in the current
	// directory.
	if *getD || len(pkgPatterns) == 0 {
		return
	}
	work.BuildInit()
	pkgs := load.PackagesForBuild(ctx, pkgPatterns)
	work.InstallPackages(ctx, pkgPatterns, pkgs)
}

// runQueries looks up modules at target versions in parallel. Results will be
// cached. If the same module is referenced by multiple queries at different
// versions (including earlier queries in the modOnly map), an error will be
// reported. A map from module paths to queries is returned, which includes
// queries and modOnly.
func runQueries(ctx context.Context, cache map[querySpec]*query, queries []*query, modOnly map[string]*query) map[string]*query {

	runQuery := func(q *query) {
		if q.vers == "none" {
			// Wait for downgrade step.
			q.m = module.Version{Path: q.path, Version: "none"}
			return
		}
		m, err := getQuery(ctx, q.path, q.vers, q.prevM, q.forceModulePath)
		if err != nil {
			base.Errorf("go get %s: %v", q.arg, err)
		}
		q.m = m
	}

	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))
	for _, q := range queries {
		if cached := cache[q.querySpec]; cached != nil {
			*q = *cached
		} else {
			sem <- token{}
			go func(q *query) {
				runQuery(q)
				<-sem
			}(q)
		}
	}

	// Fill semaphore channel to wait for goroutines to finish.
	for n := cap(sem); n > 0; n-- {
		sem <- token{}
	}

	// Add to cache after concurrent section to avoid races...
	for _, q := range queries {
		cache[q.querySpec] = q
	}

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

// getQuery evaluates the given (package or module) path and version
// to determine the underlying module version being requested.
// If forceModulePath is set, getQuery must interpret path
// as a module path.
func getQuery(ctx context.Context, path, vers string, prevM module.Version, forceModulePath bool) (module.Version, error) {
	if (prevM.Version != "") != forceModulePath {
		// We resolve package patterns by calling QueryPattern, which does not
		// accept a previous version and therefore cannot take it into account for
		// the "latest" or "patch" queries.
		// If we are resolving a package path or pattern, the caller has already
		// resolved any existing packages to their containing module(s), and
		// will set both prevM.Version and forceModulePath for those modules.
		// The only remaining package patterns are those that are not already
		// provided by the build list, which are indicated by
		// an empty prevM.Version.
		base.Fatalf("go get: internal error: prevM may be set if and only if forceModulePath is set")
	}

	// If vers is a query like "latest", we should ignore retracted and excluded
	// versions. If vers refers to a specific version or commit like "v1.0.0"
	// or "master", we should only ignore excluded versions.
	allowed := modload.CheckAllowed
	if modload.IsRevisionQuery(vers) {
		allowed = modload.CheckExclusions
	}

	// If the query must be a module path, try only that module path.
	if forceModulePath {
		if path == modload.Target.Path {
			if vers != "latest" {
				return module.Version{}, fmt.Errorf("can't get a specific version of the main module")
			}
		}

		info, err := modload.Query(ctx, path, vers, prevM.Version, allowed)
		if err == nil {
			if info.Version != vers && info.Version != prevM.Version {
				logOncef("go: %s %s => %s", path, vers, info.Version)
			}
			return module.Version{Path: path, Version: info.Version}, nil
		}

		// If the query was "upgrade" or "patch" and the current version has been
		// replaced, check to see whether the error was for that same version:
		// if so, the version was probably replaced because it is invalid,
		// and we should keep that replacement without complaining.
		if vers == "upgrade" || vers == "patch" {
			var vErr *module.InvalidVersionError
			if errors.As(err, &vErr) && vErr.Version == prevM.Version && modload.Replacement(prevM).Path != "" {
				return prevM, nil
			}
		}

		return module.Version{}, err
	}

	// If the query may be either a package or a module, try it as a package path.
	// If it turns out to only exist as a module, we can detect the resulting
	// PackageNotInModuleError and avoid a second round-trip through (potentially)
	// all of the configured proxies.
	results, err := modload.QueryPattern(ctx, path, vers, allowed)
	if err != nil {
		// If the path doesn't contain a wildcard, check whether it was actually a
		// module path instead. If so, return that.
		if !strings.Contains(path, "...") {
			var modErr *modload.PackageNotInModuleError
			if errors.As(err, &modErr) && modErr.Mod.Path == path {
				if modErr.Mod.Version != vers {
					logOncef("go: %s %s => %s", path, vers, modErr.Mod.Version)
				}
				return modErr.Mod, nil
			}
		}

		return module.Version{}, err
	}

	m := results[0].Mod
	if m.Path != path {
		logOncef("go: found %s in %s %s", path, m.Path, m.Version)
	} else if m.Version != vers {
		logOncef("go: %s %s => %s", path, vers, m.Version)
	}
	return m, nil
}

// reportRetractions prints warnings if any modules in the build list are
// retracted.
func reportRetractions(ctx context.Context) {
	// Query for retractions of modules in the build list.
	// Use modload.ListModules, since that provides information in the same format
	// as 'go list -m'. Don't query for "all", since that's not allowed outside a
	// module.
	buildList := modload.LoadedModules()
	args := make([]string, 0, len(buildList))
	for _, m := range buildList {
		if m.Version == "" {
			// main module or dummy target module
			continue
		}
		args = append(args, m.Path+"@"+m.Version)
	}
	listU := false
	listVersions := false
	listRetractions := true
	mods := modload.ListModules(ctx, args, listU, listVersions, listRetractions)
	retractPath := ""
	for _, mod := range mods {
		if len(mod.Retracted) > 0 {
			if retractPath == "" {
				retractPath = mod.Path
			} else {
				retractPath = "<module>"
			}
			rationale := modload.ShortRetractionRationale(mod.Retracted[0])
			logOncef("go: warning: %s@%s is retracted: %s", mod.Path, mod.Version, rationale)
		}
	}
	if modload.HasModRoot() && retractPath != "" {
		logOncef("go: run 'go get %s@latest' to switch to the latest unretracted version", retractPath)
	}
}

var loggedLines sync.Map

func logOncef(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	if _, dup := loggedLines.LoadOrStore(msg, true); !dup {
		fmt.Fprintln(os.Stderr, msg)
	}
}
