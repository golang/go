// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package modget implements the module-aware “go get” command.
package modget

// The arguments to 'go get' are patterns with optional version queries, with
// the version queries defaulting to "upgrade".
//
// The patterns are normally interpreted as package patterns. However, if a
// pattern cannot match a package, it is instead interpreted as a *module*
// pattern. For version queries such as "upgrade" and "patch" that depend on the
// selected version of a module (or of the module containing a package),
// whether a pattern denotes a package or module may change as updates are
// applied (see the example in mod_get_patchmod.txt).
//
// There are a few other ambiguous cases to resolve, too. A package can exist in
// two different modules at the same version: for example, the package
// example.com/foo might be found in module example.com and also in module
// example.com/foo, and those modules may have independent v0.1.0 tags — so the
// input 'example.com/foo@v0.1.0' could syntactically refer to the variant of
// the package loaded from either module! (See mod_get_ambiguous_pkg.txt.)
// If the argument is ambiguous, the user can often disambiguate by specifying
// explicit versions for *all* of the potential module paths involved.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/imports"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"
	"cmd/go/internal/search"
	"cmd/go/internal/toolchain"
	"cmd/go/internal/work"
	"cmd/internal/par"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
)

var CmdGet = &base.Command{
	// Note: flags below are listed explicitly because they're the most common.
	// Do not send CLs removing them because they're covered by [get flags].
	UsageLine: "go get [-t] [-u] [-tool] [build flags] [packages]",
	Short:     "add dependencies to current module and install them",
	Long: `
Get resolves its command-line arguments to packages at specific module versions,
updates go.mod to require those versions, and downloads source code into the
module cache.

To add a dependency for a package or upgrade it to its latest version:

	go get example.com/pkg

To upgrade or downgrade a package to a specific version:

	go get example.com/pkg@v1.2.3

To remove a dependency on a module and downgrade modules that require it:

	go get example.com/mod@none

To upgrade the minimum required Go version to the latest released Go version:

	go get go@latest

To upgrade the Go toolchain to the latest patch release of the current Go toolchain:

	go get toolchain@patch

See https://golang.org/ref/mod#go-get for details.

In earlier versions of Go, 'go get' was used to build and install packages.
Now, 'go get' is dedicated to adjusting dependencies in go.mod. 'go install'
may be used to build and install commands instead. When a version is specified,
'go install' runs in module-aware mode and ignores the go.mod file in the
current directory. For example:

	go install example.com/pkg@v1.2.3
	go install example.com/pkg@latest

See 'go help install' or https://golang.org/ref/mod#go-install for details.

'go get' accepts the following flags.

The -t flag instructs get to consider modules needed to build tests of
packages specified on the command line.

The -u flag instructs get to update modules providing dependencies
of packages named on the command line to use newer minor or patch
releases when available.

The -u=patch flag (not -u patch) also instructs get to update dependencies,
but changes the default to select patch releases.

When the -t and -u flags are used together, get will update
test dependencies as well.

The -tool flag instructs go to add a matching tool line to go.mod for each
listed package. If -tool is used with @none, the line will be removed.

The -x flag prints commands as they are executed. This is useful for
debugging version control commands when a module is downloaded directly
from a repository.

For more about build flags, see 'go help build'.

For more about modules, see https://golang.org/ref/mod.

For more about using 'go get' to update the minimum Go version and
suggested Go toolchain, see https://go.dev/doc/toolchain.

For more about specifying packages, see 'go help packages'.

This text describes the behavior of get using modules to manage source
code and dependencies. If instead the go command is running in GOPATH
mode, the details of get's flags and effects change, as does 'go help get'.
See 'go help gopath-get'.

See also: go build, go install, go clean, go mod.
	`,
}

var HelpVCS = &base.Command{
	UsageLine: "vcs",
	Short:     "controlling version control with GOVCS",
	Long: `
The 'go get' command can run version control commands like git
to download imported code. This functionality is critical to the decentralized
Go package ecosystem, in which code can be imported from any server,
but it is also a potential security problem, if a malicious server finds a
way to cause the invoked version control command to run unintended code.

To balance the functionality and security concerns, the 'go get' command
by default will only use git and hg to download code from public servers.
But it will use any known version control system (bzr, fossil, git, hg, svn)
to download code from private servers, defined as those hosting packages
matching the GOPRIVATE variable (see 'go help private'). The rationale behind
allowing only Git and Mercurial is that these two systems have had the most
attention to issues of being run as clients of untrusted servers. In contrast,
Bazaar, Fossil, and Subversion have primarily been used in trusted,
authenticated environments and are not as well scrutinized as attack surfaces.

The version control command restrictions only apply when using direct version
control access to download code. When downloading modules from a proxy,
'go get' uses the proxy protocol instead, which is always permitted.
By default, the 'go get' command uses the Go module mirror (proxy.golang.org)
for public packages and only falls back to version control for private
packages or when the mirror refuses to serve a public package (typically for
legal reasons). Therefore, clients can still access public code served from
Bazaar, Fossil, or Subversion repositories by default, because those downloads
use the Go module mirror, which takes on the security risk of running the
version control commands using a custom sandbox.

The GOVCS variable can be used to change the allowed version control systems
for specific packages (identified by a module or import path).
The GOVCS variable applies when building package in both module-aware mode
and GOPATH mode. When using modules, the patterns match against the module path.
When using GOPATH, the patterns match against the import path corresponding to
the root of the version control repository.

The general form of the GOVCS setting is a comma-separated list of
pattern:vcslist rules. The pattern is a glob pattern that must match
one or more leading elements of the module or import path. The vcslist
is a pipe-separated list of allowed version control commands, or "all"
to allow use of any known command, or "off" to disallow all commands.
Note that if a module matches a pattern with vcslist "off", it may still be
downloaded if the origin server uses the "mod" scheme, which instructs the
go command to download the module using the GOPROXY protocol.
The earliest matching pattern in the list applies, even if later patterns
might also match.

For example, consider:

	GOVCS=github.com:git,evil.com:off,*:git|hg

With this setting, code with a module or import path beginning with
github.com/ can only use git; paths on evil.com cannot use any version
control command, and all other paths (* matches everything) can use
only git or hg.

The special patterns "public" and "private" match public and private
module or import paths. A path is private if it matches the GOPRIVATE
variable; otherwise it is public.

If no rules in the GOVCS variable match a particular module or import path,
the 'go get' command applies its default rule, which can now be summarized
in GOVCS notation as 'public:git|hg,private:all'.

To allow unfettered use of any version control system for any package, use:

	GOVCS=*:all

To disable all use of version control, use:

	GOVCS=*:off

The 'go env -w' command (see 'go help env') can be used to set the GOVCS
variable for future go command invocations.
`,
}

var (
	getD        dFlag
	getF        = CmdGet.Flag.Bool("f", false, "")
	getFix      = CmdGet.Flag.Bool("fix", false, "")
	getM        = CmdGet.Flag.Bool("m", false, "")
	getT        = CmdGet.Flag.Bool("t", false, "")
	getU        upgradeFlag
	getTool     = CmdGet.Flag.Bool("tool", false, "")
	getInsecure = CmdGet.Flag.Bool("insecure", false, "")
)

// upgradeFlag is a custom flag.Value for -u.
type upgradeFlag struct {
	rawVersion string
	version    string
}

func (*upgradeFlag) IsBoolFlag() bool { return true } // allow -u

func (v *upgradeFlag) Set(s string) error {
	if s == "false" {
		v.version = ""
		v.rawVersion = ""
	} else if s == "true" {
		v.version = "upgrade"
		v.rawVersion = ""
	} else {
		v.version = s
		v.rawVersion = s
	}
	return nil
}

func (v *upgradeFlag) String() string { return "" }

// dFlag is a custom flag.Value for the deprecated -d flag
// which will be used to provide warnings or errors if -d
// is provided.
type dFlag struct {
	value bool
	set   bool
}

func (v *dFlag) IsBoolFlag() bool { return true }

func (v *dFlag) Set(s string) error {
	v.set = true
	value, err := strconv.ParseBool(s)
	if err != nil {
		err = errors.New("parse error")
	}
	v.value = value
	return err
}

func (b *dFlag) String() string { return "" }

func init() {
	work.AddBuildFlags(CmdGet, work.OmitModFlag)
	CmdGet.Run = runGet // break init loop
	CmdGet.Flag.Var(&getD, "d", "")
	CmdGet.Flag.Var(&getU, "u", "")
}

func runGet(ctx context.Context, cmd *base.Command, args []string) {
	switch getU.version {
	case "", "upgrade", "patch":
		// ok
	default:
		base.Fatalf("go: unknown upgrade flag -u=%s", getU.rawVersion)
	}
	if getD.set {
		if !getD.value {
			base.Fatalf("go: -d flag may not be set to false")
		}
		fmt.Fprintf(os.Stderr, "go: -d flag is deprecated. -d=true is a no-op\n")
	}
	if *getF {
		fmt.Fprintf(os.Stderr, "go: -f flag is a no-op\n")
	}
	if *getFix {
		fmt.Fprintf(os.Stderr, "go: -fix flag is a no-op\n")
	}
	if *getM {
		base.Fatalf("go: -m flag is no longer supported")
	}
	if *getInsecure {
		base.Fatalf("go: -insecure flag is no longer supported; use GOINSECURE instead")
	}

	modload.ForceUseModules = true

	// Do not allow any updating of go.mod until we've applied
	// all the requested changes and checked that the result matches
	// what was requested.
	modload.ExplicitWriteGoMod = true

	// Allow looking up modules for import paths when outside of a module.
	// 'go get' is expected to do this, unlike other commands.
	modload.AllowMissingModuleImports()

	// 'go get' no longer builds or installs packages, so there's nothing to do
	// if there's no go.mod file.
	// TODO(#40775): make modload.Init return ErrNoModRoot instead of exiting.
	// We could handle that here by printing a different message.
	modload.Init()
	if !modload.HasModRoot() {
		base.Fatalf("go: go.mod file not found in current directory or any parent directory.\n" +
			"\t'go get' is no longer supported outside a module.\n" +
			"\tTo build and install a command, use 'go install' with a version,\n" +
			"\tlike 'go install example.com/cmd@latest'\n" +
			"\tFor more information, see https://golang.org/doc/go-get-install-deprecation\n" +
			"\tor run 'go help get' or 'go help install'.")
	}

	dropToolchain, queries := parseArgs(ctx, args)
	opts := modload.WriteOpts{
		DropToolchain: dropToolchain,
	}
	for _, q := range queries {
		if q.pattern == "toolchain" {
			opts.ExplicitToolchain = true
		}
	}

	r := newResolver(ctx, queries)
	r.performLocalQueries(ctx)
	r.performPathQueries(ctx)
	r.performToolQueries(ctx)

	for {
		r.performWildcardQueries(ctx)
		r.performPatternAllQueries(ctx)

		if changed := r.resolveQueries(ctx, queries); changed {
			// 'go get' arguments can be (and often are) package patterns rather than
			// (just) modules. A package can be provided by any module with a prefix
			// of its import path, and a wildcard can even match packages in modules
			// with totally different paths. Because of these effects, and because any
			// change to the selected version of a module can bring in entirely new
			// module paths as dependencies, we need to reissue queries whenever we
			// change the build list.
			//
			// The result of any version query for a given module — even "upgrade" or
			// "patch" — is always relative to the build list at the start of
			// the 'go get' command, not an intermediate state, and is therefore
			// deterministic and therefore cacheable, and the constraints on the
			// selected version of each module can only narrow as we iterate.
			//
			// "all" is functionally very similar to a wildcard pattern. The set of
			// packages imported by the main module does not change, and the query
			// result for the module containing each such package also does not change
			// (it is always relative to the initial build list, before applying
			// queries). So the only way that the result of an "all" query can change
			// is if some matching package moves from one module in the build list
			// to another, which should not happen very often.
			continue
		}

		// When we load imports, we detect the following conditions:
		//
		// - missing transitive dependencies that need to be resolved from outside the
		//   current build list (note that these may add new matches for existing
		//   pattern queries!)
		//
		// - transitive dependencies that didn't match any other query,
		//   but need to be upgraded due to the -u flag
		//
		// - ambiguous import errors.
		//   TODO(#27899): Try to resolve ambiguous import errors automatically.
		upgrades := r.findAndUpgradeImports(ctx, queries)
		if changed := r.applyUpgrades(ctx, upgrades); changed {
			continue
		}

		r.findMissingWildcards(ctx)
		if changed := r.resolveQueries(ctx, r.wildcardQueries); changed {
			continue
		}

		break
	}

	r.checkWildcardVersions(ctx)

	var pkgPatterns []string
	for _, q := range queries {
		if q.matchesPackages {
			pkgPatterns = append(pkgPatterns, q.pattern)
		}
	}
	r.checkPackageProblems(ctx, pkgPatterns)

	if *getTool {
		updateTools(ctx, queries, &opts)
	}

	// Everything succeeded. Update go.mod.
	oldReqs := reqsFromGoMod(modload.ModFile())

	if err := modload.WriteGoMod(ctx, opts); err != nil {
		// A TooNewError can happen for 'go get go@newversion'
		// when all the required modules are old enough
		// but the command line is not.
		// TODO(bcmills): modload.EditBuildList should catch this instead,
		// and then this can be changed to base.Fatal(err).
		toolchain.SwitchOrFatal(ctx, err)
	}

	newReqs := reqsFromGoMod(modload.ModFile())
	r.reportChanges(oldReqs, newReqs)

	if gowork := modload.FindGoWork(base.Cwd()); gowork != "" {
		wf, err := modload.ReadWorkFile(gowork)
		if err == nil && modload.UpdateWorkGoVersion(wf, modload.MainModules.GoVersion()) {
			modload.WriteWorkFile(gowork, wf)
		}
	}
}

func updateTools(ctx context.Context, queries []*query, opts *modload.WriteOpts) {
	pkgOpts := modload.PackageOpts{
		VendorModulesInGOROOTSrc: true,
		LoadTests:                *getT,
		ResolveMissingImports:    false,
		AllowErrors:              true,
		SilenceNoGoErrors:        true,
	}
	patterns := []string{}
	for _, q := range queries {
		if search.IsMetaPackage(q.pattern) || q.pattern == "toolchain" {
			base.Fatalf("go: go get -tool does not work with \"%s\".", q.pattern)
		}
		patterns = append(patterns, q.pattern)
	}

	matches, _ := modload.LoadPackages(ctx, pkgOpts, patterns...)
	for i, m := range matches {
		if queries[i].version == "none" {
			opts.DropTools = append(opts.DropTools, m.Pkgs...)
		} else {
			opts.AddTools = append(opts.DropTools, m.Pkgs...)
		}
	}
}

// parseArgs parses command-line arguments and reports errors.
//
// The command-line arguments are of the form path@version or simply path, with
// implicit @upgrade. path@none is "downgrade away".
func parseArgs(ctx context.Context, rawArgs []string) (dropToolchain bool, queries []*query) {
	defer base.ExitIfErrors()

	for _, arg := range search.CleanPatterns(rawArgs) {
		q, err := newQuery(arg)
		if err != nil {
			base.Error(err)
			continue
		}

		if q.version == "none" {
			switch q.pattern {
			case "go":
				base.Errorf("go: cannot use go@none")
				continue
			case "toolchain":
				dropToolchain = true
				continue
			}
		}

		// If there were no arguments, CleanPatterns returns ".". Set the raw
		// string back to "" for better errors.
		if len(rawArgs) == 0 {
			q.raw = ""
		}

		// Guard against 'go get x.go', a common mistake.
		// Note that package and module paths may end with '.go', so only print an error
		// if the argument has no version and either has no slash or refers to an existing file.
		if strings.HasSuffix(q.raw, ".go") && q.rawVersion == "" {
			if !strings.Contains(q.raw, "/") {
				base.Errorf("go: %s: arguments must be package or module paths", q.raw)
				continue
			}
			if fi, err := os.Stat(q.raw); err == nil && !fi.IsDir() {
				base.Errorf("go: %s exists as a file, but 'go get' requires package arguments", q.raw)
				continue
			}
		}

		queries = append(queries, q)
	}

	return dropToolchain, queries
}

type resolver struct {
	localQueries      []*query // queries for absolute or relative paths
	pathQueries       []*query // package path literal queries in original order
	wildcardQueries   []*query // path wildcard queries in original order
	patternAllQueries []*query // queries with the pattern "all"
	toolQueries       []*query // queries with the pattern "tool"

	// Indexed "none" queries. These are also included in the slices above;
	// they are indexed here to speed up noneForPath.
	nonesByPath   map[string]*query // path-literal "@none" queries indexed by path
	wildcardNones []*query          // wildcard "@none" queries

	// resolvedVersion maps each module path to the version of that module that
	// must be selected in the final build list, along with the first query
	// that resolved the module to that version (the “reason”).
	resolvedVersion map[string]versionReason

	buildList        []module.Version
	buildListVersion map[string]string // index of buildList (module path → version)

	initialVersion map[string]string // index of the initial build list at the start of 'go get'

	missing []pathSet // candidates for missing transitive dependencies

	work *par.Queue

	matchInModuleCache par.ErrCache[matchInModuleKey, []string]
}

type versionReason struct {
	version string
	reason  *query
}

type matchInModuleKey struct {
	pattern string
	m       module.Version
}

func newResolver(ctx context.Context, queries []*query) *resolver {
	// LoadModGraph also sets modload.Target, which is needed by various resolver
	// methods.
	mg, err := modload.LoadModGraph(ctx, "")
	if err != nil {
		toolchain.SwitchOrFatal(ctx, err)
	}

	buildList := mg.BuildList()
	initialVersion := make(map[string]string, len(buildList))
	for _, m := range buildList {
		initialVersion[m.Path] = m.Version
	}

	r := &resolver{
		work:             par.NewQueue(runtime.GOMAXPROCS(0)),
		resolvedVersion:  map[string]versionReason{},
		buildList:        buildList,
		buildListVersion: initialVersion,
		initialVersion:   initialVersion,
		nonesByPath:      map[string]*query{},
	}

	for _, q := range queries {
		if q.pattern == "all" {
			r.patternAllQueries = append(r.patternAllQueries, q)
		} else if q.pattern == "tool" {
			r.toolQueries = append(r.toolQueries, q)
		} else if q.patternIsLocal {
			r.localQueries = append(r.localQueries, q)
		} else if q.isWildcard() {
			r.wildcardQueries = append(r.wildcardQueries, q)
		} else {
			r.pathQueries = append(r.pathQueries, q)
		}

		if q.version == "none" {
			// Index "none" queries to make noneForPath more efficient.
			if q.isWildcard() {
				r.wildcardNones = append(r.wildcardNones, q)
			} else {
				// All "<path>@none" queries for the same path are identical; we only
				// need to index one copy.
				r.nonesByPath[q.pattern] = q
			}
		}
	}

	return r
}

// initialSelected returns the version of the module with the given path that
// was selected at the start of this 'go get' invocation.
func (r *resolver) initialSelected(mPath string) (version string) {
	v, ok := r.initialVersion[mPath]
	if !ok {
		return "none"
	}
	return v
}

// selected returns the version of the module with the given path that is
// selected in the resolver's current build list.
func (r *resolver) selected(mPath string) (version string) {
	v, ok := r.buildListVersion[mPath]
	if !ok {
		return "none"
	}
	return v
}

// noneForPath returns a "none" query matching the given module path,
// or found == false if no such query exists.
func (r *resolver) noneForPath(mPath string) (nq *query, found bool) {
	if nq = r.nonesByPath[mPath]; nq != nil {
		return nq, true
	}
	for _, nq := range r.wildcardNones {
		if nq.matchesPath(mPath) {
			return nq, true
		}
	}
	return nil, false
}

// queryModule wraps modload.Query, substituting r.checkAllowedOr to decide
// allowed versions.
func (r *resolver) queryModule(ctx context.Context, mPath, query string, selected func(string) string) (module.Version, error) {
	current := r.initialSelected(mPath)
	rev, err := modload.Query(ctx, mPath, query, current, r.checkAllowedOr(query, selected))
	if err != nil {
		return module.Version{}, err
	}
	return module.Version{Path: mPath, Version: rev.Version}, nil
}

// queryPackages wraps modload.QueryPackage, substituting r.checkAllowedOr to
// decide allowed versions.
func (r *resolver) queryPackages(ctx context.Context, pattern, query string, selected func(string) string) (pkgMods []module.Version, err error) {
	results, err := modload.QueryPackages(ctx, pattern, query, selected, r.checkAllowedOr(query, selected))
	if len(results) > 0 {
		pkgMods = make([]module.Version, 0, len(results))
		for _, qr := range results {
			pkgMods = append(pkgMods, qr.Mod)
		}
	}
	return pkgMods, err
}

// queryPattern wraps modload.QueryPattern, substituting r.checkAllowedOr to
// decide allowed versions.
func (r *resolver) queryPattern(ctx context.Context, pattern, query string, selected func(string) string) (pkgMods []module.Version, mod module.Version, err error) {
	results, modOnly, err := modload.QueryPattern(ctx, pattern, query, selected, r.checkAllowedOr(query, selected))
	if len(results) > 0 {
		pkgMods = make([]module.Version, 0, len(results))
		for _, qr := range results {
			pkgMods = append(pkgMods, qr.Mod)
		}
	}
	if modOnly != nil {
		mod = modOnly.Mod
	}
	return pkgMods, mod, err
}

// checkAllowedOr is like modload.CheckAllowed, but it always allows the requested
// and current versions (even if they are retracted or otherwise excluded).
func (r *resolver) checkAllowedOr(requested string, selected func(string) string) modload.AllowedFunc {
	return func(ctx context.Context, m module.Version) error {
		if m.Version == requested {
			return modload.CheckExclusions(ctx, m)
		}
		if (requested == "upgrade" || requested == "patch") && m.Version == selected(m.Path) {
			return nil
		}
		return modload.CheckAllowed(ctx, m)
	}
}

// matchInModule is a caching wrapper around modload.MatchInModule.
func (r *resolver) matchInModule(ctx context.Context, pattern string, m module.Version) (packages []string, err error) {
	return r.matchInModuleCache.Do(matchInModuleKey{pattern, m}, func() ([]string, error) {
		match := modload.MatchInModule(ctx, pattern, m, imports.AnyTags())
		if len(match.Errs) > 0 {
			return match.Pkgs, match.Errs[0]
		}
		return match.Pkgs, nil
	})
}

// queryNone adds a candidate set to q for each module matching q.pattern.
// Each candidate set has only one possible module version: the matched
// module at version "none".
//
// We interpret arguments to 'go get' as packages first, and fall back to
// modules second. However, no module exists at version "none", and therefore no
// package exists at that version either: we know that the argument cannot match
// any packages, and thus it must match modules instead.
func (r *resolver) queryNone(ctx context.Context, q *query) {
	if search.IsMetaPackage(q.pattern) {
		panic(fmt.Sprintf("internal error: queryNone called with pattern %q", q.pattern))
	}

	if !q.isWildcard() {
		q.pathOnce(q.pattern, func() pathSet {
			hasModRoot := modload.HasModRoot()
			if hasModRoot && modload.MainModules.Contains(q.pattern) {
				v := module.Version{Path: q.pattern}
				// The user has explicitly requested to downgrade their own module to
				// version "none". This is not an entirely unreasonable request: it
				// could plausibly mean “downgrade away everything that depends on any
				// explicit version of the main module”, or “downgrade away the
				// package with the same path as the main module, found in a module
				// with a prefix of the main module's path”.
				//
				// However, neither of those behaviors would be consistent with the
				// plain meaning of the query. To try to reduce confusion, reject the
				// query explicitly.
				return errSet(&modload.QueryMatchesMainModulesError{MainModules: []module.Version{v}, Pattern: q.pattern, Query: q.version})
			}

			return pathSet{mod: module.Version{Path: q.pattern, Version: "none"}}
		})
	}

	for _, curM := range r.buildList {
		if !q.matchesPath(curM.Path) {
			continue
		}
		q.pathOnce(curM.Path, func() pathSet {
			if modload.HasModRoot() && curM.Version == "" && modload.MainModules.Contains(curM.Path) {
				return errSet(&modload.QueryMatchesMainModulesError{MainModules: []module.Version{curM}, Pattern: q.pattern, Query: q.version})
			}
			return pathSet{mod: module.Version{Path: curM.Path, Version: "none"}}
		})
	}
}

func (r *resolver) performLocalQueries(ctx context.Context) {
	for _, q := range r.localQueries {
		q.pathOnce(q.pattern, func() pathSet {
			absDetail := ""
			if !filepath.IsAbs(q.pattern) {
				if absPath, err := filepath.Abs(q.pattern); err == nil {
					absDetail = fmt.Sprintf(" (%s)", absPath)
				}
			}

			// Absolute paths like C:\foo and relative paths like ../foo... are
			// restricted to matching packages in the main module.
			pkgPattern, mainModule := modload.MainModules.DirImportPath(ctx, q.pattern)
			if pkgPattern == "." {
				modload.MustHaveModRoot()
				versions := modload.MainModules.Versions()
				modRoots := make([]string, 0, len(versions))
				for _, m := range versions {
					modRoots = append(modRoots, modload.MainModules.ModRoot(m))
				}
				var plural string
				if len(modRoots) != 1 {
					plural = "s"
				}
				return errSet(fmt.Errorf("%s%s is not within module%s rooted at %s", q.pattern, absDetail, plural, strings.Join(modRoots, ", ")))
			}

			match := modload.MatchInModule(ctx, pkgPattern, mainModule, imports.AnyTags())
			if len(match.Errs) > 0 {
				return pathSet{err: match.Errs[0]}
			}

			if len(match.Pkgs) == 0 {
				if q.raw == "" || q.raw == "." {
					return errSet(fmt.Errorf("no package to get in current directory"))
				}
				if !q.isWildcard() {
					modload.MustHaveModRoot()
					return errSet(fmt.Errorf("%s%s is not a package in module rooted at %s", q.pattern, absDetail, modload.MainModules.ModRoot(mainModule)))
				}
				search.WarnUnmatched([]*search.Match{match})
				return pathSet{}
			}

			return pathSet{pkgMods: []module.Version{mainModule}}
		})
	}
}

// performWildcardQueries populates the candidates for each query whose pattern
// is a wildcard.
//
// The candidates for a given module path matching (or containing a package
// matching) a wildcard query depend only on the initial build list, but the set
// of modules may be expanded by other queries, so wildcard queries need to be
// re-evaluated whenever a potentially-matching module path is added to the
// build list.
func (r *resolver) performWildcardQueries(ctx context.Context) {
	for _, q := range r.wildcardQueries {
		q := q
		r.work.Add(func() {
			if q.version == "none" {
				r.queryNone(ctx, q)
			} else {
				r.queryWildcard(ctx, q)
			}
		})
	}
	<-r.work.Idle()
}

// queryWildcard adds a candidate set to q for each module for which:
//   - some version of the module is already in the build list, and
//   - that module exists at some version matching q.version, and
//   - either the module path itself matches q.pattern, or some package within
//     the module at q.version matches q.pattern.
func (r *resolver) queryWildcard(ctx context.Context, q *query) {
	// For wildcard patterns, modload.QueryPattern only identifies modules
	// matching the prefix of the path before the wildcard. However, the build
	// list may already contain other modules with matching packages, and we
	// should consider those modules to satisfy the query too.
	// We want to match any packages in existing dependencies, but we only want to
	// resolve new dependencies if nothing else turns up.
	for _, curM := range r.buildList {
		if !q.canMatchInModule(curM.Path) {
			continue
		}
		q.pathOnce(curM.Path, func() pathSet {
			if _, hit := r.noneForPath(curM.Path); hit {
				// This module is being removed, so it will no longer be in the build list
				// (and thus will no longer match the pattern).
				return pathSet{}
			}

			if modload.MainModules.Contains(curM.Path) && !versionOkForMainModule(q.version) {
				if q.matchesPath(curM.Path) {
					return errSet(&modload.QueryMatchesMainModulesError{
						MainModules: []module.Version{curM},
						Pattern:     q.pattern,
						Query:       q.version,
					})
				}

				packages, err := r.matchInModule(ctx, q.pattern, curM)
				if err != nil {
					return errSet(err)
				}
				if len(packages) > 0 {
					return errSet(&modload.QueryMatchesPackagesInMainModuleError{
						Pattern:  q.pattern,
						Query:    q.version,
						Packages: packages,
					})
				}

				return r.tryWildcard(ctx, q, curM)
			}

			m, err := r.queryModule(ctx, curM.Path, q.version, r.initialSelected)
			if err != nil {
				if !isNoSuchModuleVersion(err) {
					// We can't tell whether a matching version exists.
					return errSet(err)
				}
				// There is no version of curM.Path matching the query.

				// We haven't checked whether curM contains any matching packages at its
				// currently-selected version, or whether curM.Path itself matches q. If
				// either of those conditions holds, *and* no other query changes the
				// selected version of curM, then we will fail in checkWildcardVersions.
				// (This could be an error, but it's too soon to tell.)
				//
				// However, even then the transitive requirements of some other query
				// may downgrade this module out of the build list entirely, in which
				// case the pattern will no longer include it and it won't be an error.
				//
				// Either way, punt on the query rather than erroring out just yet.
				return pathSet{}
			}

			return r.tryWildcard(ctx, q, m)
		})
	}

	// Even if no modules matched, we shouldn't query for a new module to provide
	// the pattern yet: some other query may yet induce a new requirement that
	// will match the wildcard. Instead, we'll check in findMissingWildcards.
}

// tryWildcard returns a pathSet for module m matching query q.
// If m does not actually match q, tryWildcard returns an empty pathSet.
func (r *resolver) tryWildcard(ctx context.Context, q *query, m module.Version) pathSet {
	mMatches := q.matchesPath(m.Path)
	packages, err := r.matchInModule(ctx, q.pattern, m)
	if err != nil {
		return errSet(err)
	}
	if len(packages) > 0 {
		return pathSet{pkgMods: []module.Version{m}}
	}
	if mMatches {
		return pathSet{mod: m}
	}
	return pathSet{}
}

// findMissingWildcards adds a candidate set for each query in r.wildcardQueries
// that has not yet resolved to any version containing packages.
func (r *resolver) findMissingWildcards(ctx context.Context) {
	for _, q := range r.wildcardQueries {
		if q.version == "none" || q.matchesPackages {
			continue // q is not “missing”
		}
		r.work.Add(func() {
			q.pathOnce(q.pattern, func() pathSet {
				pkgMods, mod, err := r.queryPattern(ctx, q.pattern, q.version, r.initialSelected)
				if err != nil {
					if isNoSuchPackageVersion(err) && len(q.resolved) > 0 {
						// q already resolved one or more modules but matches no packages.
						// That's ok: this pattern is just a module pattern, and we don't
						// need to add any more modules to satisfy it.
						return pathSet{}
					}
					return errSet(err)
				}

				return pathSet{pkgMods: pkgMods, mod: mod}
			})
		})
	}
	<-r.work.Idle()
}

// checkWildcardVersions reports an error if any module in the build list has a
// path (or contains a package) matching a query with a wildcard pattern, but
// has a selected version that does *not* match the query.
func (r *resolver) checkWildcardVersions(ctx context.Context) {
	defer base.ExitIfErrors()

	for _, q := range r.wildcardQueries {
		for _, curM := range r.buildList {
			if !q.canMatchInModule(curM.Path) {
				continue
			}
			if !q.matchesPath(curM.Path) {
				packages, err := r.matchInModule(ctx, q.pattern, curM)
				if len(packages) == 0 {
					if err != nil {
						reportError(q, err)
					}
					continue // curM is not relevant to q.
				}
			}

			rev, err := r.queryModule(ctx, curM.Path, q.version, r.initialSelected)
			if err != nil {
				reportError(q, err)
				continue
			}
			if rev.Version == curM.Version {
				continue // curM already matches q.
			}

			if !q.matchesPath(curM.Path) {
				m := module.Version{Path: curM.Path, Version: rev.Version}
				packages, err := r.matchInModule(ctx, q.pattern, m)
				if err != nil {
					reportError(q, err)
					continue
				}
				if len(packages) == 0 {
					// curM at its original version contains a path matching q.pattern,
					// but at rev.Version it does not, so (somewhat paradoxically) if
					// we changed the version of curM it would no longer match the query.
					var version any = m
					if rev.Version != q.version {
						version = fmt.Sprintf("%s@%s (%s)", m.Path, q.version, m.Version)
					}
					reportError(q, fmt.Errorf("%v matches packages in %v but not %v: specify a different version for module %s", q, curM, version, m.Path))
					continue
				}
			}

			// Since queryModule succeeded and either curM or one of the packages it
			// contains matches q.pattern, we should have either selected the version
			// of curM matching q, or reported a conflict error (and exited).
			// If we're still here and the version doesn't match,
			// something has gone very wrong.
			reportError(q, fmt.Errorf("internal error: selected %v instead of %v", curM, rev.Version))
		}
	}
}

// performPathQueries populates the candidates for each query whose pattern is
// a path literal.
//
// The candidate packages and modules for path literals depend only on the
// initial build list, not the current build list, so we only need to query path
// literals once.
func (r *resolver) performPathQueries(ctx context.Context) {
	for _, q := range r.pathQueries {
		q := q
		r.work.Add(func() {
			if q.version == "none" {
				r.queryNone(ctx, q)
			} else {
				r.queryPath(ctx, q)
			}
		})
	}
	<-r.work.Idle()
}

// queryPath adds a candidate set to q for the package with path q.pattern.
// The candidate set consists of all modules that could provide q.pattern
// and have a version matching q, plus (if it exists) the module whose path
// is itself q.pattern (at a matching version).
func (r *resolver) queryPath(ctx context.Context, q *query) {
	q.pathOnce(q.pattern, func() pathSet {
		if search.IsMetaPackage(q.pattern) || q.isWildcard() {
			panic(fmt.Sprintf("internal error: queryPath called with pattern %q", q.pattern))
		}
		if q.version == "none" {
			panic(`internal error: queryPath called with version "none"`)
		}

		if search.IsStandardImportPath(q.pattern) {
			stdOnly := module.Version{}
			packages, _ := r.matchInModule(ctx, q.pattern, stdOnly)
			if len(packages) > 0 {
				if q.rawVersion != "" {
					return errSet(fmt.Errorf("can't request explicit version %q of standard library package %s", q.version, q.pattern))
				}

				q.matchesPackages = true
				return pathSet{} // No module needed for standard library.
			}
		}

		pkgMods, mod, err := r.queryPattern(ctx, q.pattern, q.version, r.initialSelected)
		if err != nil {
			return errSet(err)
		}
		return pathSet{pkgMods: pkgMods, mod: mod}
	})
}

// performToolQueries populates the candidates for each query whose
// pattern is "tool".
func (r *resolver) performToolQueries(ctx context.Context) {
	for _, q := range r.toolQueries {
		for tool := range modload.MainModules.Tools() {
			q.pathOnce(tool, func() pathSet {
				pkgMods, err := r.queryPackages(ctx, tool, q.version, r.initialSelected)
				return pathSet{pkgMods: pkgMods, err: err}
			})
		}
	}
}

// performPatternAllQueries populates the candidates for each query whose
// pattern is "all".
//
// The candidate modules for a given package in "all" depend only on the initial
// build list, but we cannot follow the dependencies of a given package until we
// know which candidate is selected — and that selection may depend on the
// results of other queries. We need to re-evaluate the "all" queries whenever
// the module for one or more packages in "all" are resolved.
func (r *resolver) performPatternAllQueries(ctx context.Context) {
	if len(r.patternAllQueries) == 0 {
		return
	}

	findPackage := func(ctx context.Context, path string, m module.Version) (versionOk bool) {
		versionOk = true
		for _, q := range r.patternAllQueries {
			q.pathOnce(path, func() pathSet {
				pkgMods, err := r.queryPackages(ctx, path, q.version, r.initialSelected)
				if len(pkgMods) != 1 || pkgMods[0] != m {
					// There are candidates other than m for the given path, so we can't
					// be certain that m will actually be the module selected to provide
					// the package. Don't load its dependencies just yet, because they
					// might no longer be dependencies after we resolve the correct
					// version.
					versionOk = false
				}
				return pathSet{pkgMods: pkgMods, err: err}
			})
		}
		return versionOk
	}

	r.loadPackages(ctx, []string{"all"}, findPackage)

	// Since we built up the candidate lists concurrently, they may be in a
	// nondeterministic order. We want 'go get' to be fully deterministic,
	// including in which errors it chooses to report, so sort the candidates
	// into a deterministic-but-arbitrary order.
	for _, q := range r.patternAllQueries {
		sort.Slice(q.candidates, func(i, j int) bool {
			return q.candidates[i].path < q.candidates[j].path
		})
	}
}

// findAndUpgradeImports returns a pathSet for each package that is not yet
// in the build list but is transitively imported by the packages matching the
// given queries (which must already have been resolved).
//
// If the getU flag ("-u") is set, findAndUpgradeImports also returns a
// pathSet for each module that is not constrained by any other
// command-line argument and has an available matching upgrade.
func (r *resolver) findAndUpgradeImports(ctx context.Context, queries []*query) (upgrades []pathSet) {
	patterns := make([]string, 0, len(queries))
	for _, q := range queries {
		if q.matchesPackages {
			patterns = append(patterns, q.pattern)
		}
	}
	if len(patterns) == 0 {
		return nil
	}

	// mu guards concurrent writes to upgrades, which will be sorted
	// (to restore determinism) after loading.
	var mu sync.Mutex

	findPackage := func(ctx context.Context, path string, m module.Version) (versionOk bool) {
		version := "latest"
		if m.Path != "" {
			if getU.version == "" {
				// The user did not request that we upgrade transitive dependencies.
				return true
			}
			if _, ok := r.resolvedVersion[m.Path]; ok {
				// We cannot upgrade m implicitly because its version is determined by
				// an explicit pattern argument.
				return true
			}
			version = getU.version
		}

		// Unlike other queries, the "-u" flag upgrades relative to the build list
		// after applying changes so far, not the initial build list.
		// This is for two reasons:
		//
		// 	- The "-u" flag intentionally applies to transitive dependencies,
		// 	  which may not be known or even resolved in advance of applying
		// 	  other version changes.
		//
		// 	- The "-u" flag, unlike other arguments, does not cause version
		// 	  conflicts with other queries. (The other query always wins.)

		pkgMods, err := r.queryPackages(ctx, path, version, r.selected)
		for _, u := range pkgMods {
			if u == m {
				// The selected package version is already upgraded appropriately; there
				// is no need to change it.
				return true
			}
		}

		if err != nil {
			if isNoSuchPackageVersion(err) || (m.Path == "" && module.CheckPath(path) != nil) {
				// We can't find the package because it doesn't — or can't — even exist
				// in any module at the latest version. (Note that invalid module paths
				// could in general exist due to replacements, so we at least need to
				// run the query to check those.)
				//
				// There is no version change we can make to fix the package, so leave
				// it unresolved. Either some other query (perhaps a wildcard matching a
				// newly-added dependency for some other missing package) will fill in
				// the gaps, or we will report an error (with a better import stack) in
				// the final LoadPackages call.
				return true
			}
		}

		mu.Lock()
		upgrades = append(upgrades, pathSet{path: path, pkgMods: pkgMods, err: err})
		mu.Unlock()
		return false
	}

	r.loadPackages(ctx, patterns, findPackage)

	// Since we built up the candidate lists concurrently, they may be in a
	// nondeterministic order. We want 'go get' to be fully deterministic,
	// including in which errors it chooses to report, so sort the candidates
	// into a deterministic-but-arbitrary order.
	sort.Slice(upgrades, func(i, j int) bool {
		return upgrades[i].path < upgrades[j].path
	})
	return upgrades
}

// loadPackages loads the packages matching the given patterns, invoking the
// findPackage function for each package that may require a change to the
// build list.
//
// loadPackages invokes the findPackage function for each package loaded from a
// module outside the main module. If the module or version that supplies that
// package needs to be changed due to a query, findPackage may return false
// and the imports of that package will not be loaded.
//
// loadPackages also invokes the findPackage function for each imported package
// that is neither present in the standard library nor in any module in the
// build list.
func (r *resolver) loadPackages(ctx context.Context, patterns []string, findPackage func(ctx context.Context, path string, m module.Version) (versionOk bool)) {
	opts := modload.PackageOpts{
		Tags:                     imports.AnyTags(),
		VendorModulesInGOROOTSrc: true,
		LoadTests:                *getT,
		AssumeRootsImported:      true, // After 'go get foo', imports of foo should build.
		SilencePackageErrors:     true, // May be fixed by subsequent upgrades or downgrades.
		Switcher:                 new(toolchain.Switcher),
	}

	opts.AllowPackage = func(ctx context.Context, path string, m module.Version) error {
		if m.Path == "" || m.Version == "" {
			// Packages in the standard library and main modules are already at their
			// latest (and only) available versions.
			return nil
		}
		if ok := findPackage(ctx, path, m); !ok {
			return errVersionChange
		}
		return nil
	}

	_, pkgs := modload.LoadPackages(ctx, opts, patterns...)
	for _, path := range pkgs {
		const (
			parentPath  = ""
			parentIsStd = false
		)
		_, _, err := modload.Lookup(parentPath, parentIsStd, path)
		if err == nil {
			continue
		}
		if errors.Is(err, errVersionChange) {
			// We already added candidates during loading.
			continue
		}

		var (
			importMissing *modload.ImportMissingError
			ambiguous     *modload.AmbiguousImportError
		)
		if !errors.As(err, &importMissing) && !errors.As(err, &ambiguous) {
			// The package, which is a dependency of something we care about, has some
			// problem that we can't resolve with a version change.
			// Leave the error for the final LoadPackages call.
			continue
		}

		path := path
		r.work.Add(func() {
			findPackage(ctx, path, module.Version{})
		})
	}
	<-r.work.Idle()
}

// errVersionChange is a sentinel error indicating that a module's version needs
// to be updated before its dependencies can be loaded.
var errVersionChange = errors.New("version change needed")

// resolveQueries resolves candidate sets that are attached to the given
// queries and/or needed to provide the given missing-package dependencies.
//
// resolveQueries starts by resolving one module version from each
// unambiguous pathSet attached to the given queries.
//
// If no unambiguous query results in a change to the build list,
// resolveQueries revisits the ambiguous query candidates and resolves them
// arbitrarily in order to guarantee forward progress.
//
// If all pathSets are resolved without any changes to the build list,
// resolveQueries returns with changed=false.
func (r *resolver) resolveQueries(ctx context.Context, queries []*query) (changed bool) {
	defer base.ExitIfErrors()

	// Note: this is O(N²) with the number of pathSets in the worst case.
	//
	// We could perhaps get it down to O(N) if we were to index the pathSets
	// by module path, so that we only revisit a given pathSet when the
	// version of some module in its containingPackage list has been determined.
	//
	// However, N tends to be small, and most candidate sets will include only one
	// candidate module (so they will be resolved in the first iteration), so for
	// now we'll stick to the simple O(N²) approach.

	resolved := 0
	for {
		prevResolved := resolved

		// If we found modules that were too new, find the max of the required versions
		// and then try to switch to a newer toolchain.
		var sw toolchain.Switcher
		for _, q := range queries {
			for _, cs := range q.candidates {
				sw.Error(cs.err)
			}
		}
		// Only switch if we need a newer toolchain.
		// Otherwise leave the cs.err for reporting later.
		if sw.NeedSwitch() {
			sw.Switch(ctx)
			// If NeedSwitch is true and Switch returns, Switch has failed to locate a newer toolchain.
			// It printed the errors along with one more about not finding a good toolchain.
			base.Exit()
		}

		for _, q := range queries {
			unresolved := q.candidates[:0]

			for _, cs := range q.candidates {
				if cs.err != nil {
					reportError(q, cs.err)
					resolved++
					continue
				}

				filtered, isPackage, m, unique := r.disambiguate(cs)
				if !unique {
					unresolved = append(unresolved, filtered)
					continue
				}

				if m.Path == "" {
					// The query is not viable. Choose an arbitrary candidate from
					// before filtering and “resolve” it to report a conflict.
					isPackage, m = r.chooseArbitrarily(cs)
				}
				if isPackage {
					q.matchesPackages = true
				}
				r.resolve(q, m)
				resolved++
			}

			q.candidates = unresolved
		}

		base.ExitIfErrors()
		if resolved == prevResolved {
			break // No unambiguous candidate remains.
		}
	}

	if resolved > 0 {
		if changed = r.updateBuildList(ctx, nil); changed {
			// The build list has changed, so disregard any remaining ambiguous queries:
			// they might now be determined by requirements in the build list, which we
			// would prefer to use instead of arbitrary versions.
			return true
		}
	}

	// The build list will be the same on the next iteration as it was on this
	// iteration, so any ambiguous queries will remain so. In order to make
	// progress, resolve them arbitrarily but deterministically.
	//
	// If that results in conflicting versions, the user can re-run 'go get'
	// with additional explicit versions for the conflicting packages or
	// modules.
	resolvedArbitrarily := 0
	for _, q := range queries {
		for _, cs := range q.candidates {
			isPackage, m := r.chooseArbitrarily(cs)
			if isPackage {
				q.matchesPackages = true
			}
			r.resolve(q, m)
			resolvedArbitrarily++
		}
	}
	if resolvedArbitrarily > 0 {
		changed = r.updateBuildList(ctx, nil)
	}
	return changed
}

// applyUpgrades disambiguates candidate sets that are needed to upgrade (or
// provide) transitive dependencies imported by previously-resolved packages.
//
// applyUpgrades modifies the build list by adding one module version from each
// pathSet in upgrades, then downgrading (or further upgrading) those modules as
// needed to maintain any already-resolved versions of other modules.
// applyUpgrades does not mark the new versions as resolved, so they can still
// be further modified by other queries (such as wildcards).
//
// If all pathSets are resolved without any changes to the build list,
// applyUpgrades returns with changed=false.
func (r *resolver) applyUpgrades(ctx context.Context, upgrades []pathSet) (changed bool) {
	defer base.ExitIfErrors()

	// Arbitrarily add a "latest" version that provides each missing package, but
	// do not mark the version as resolved: we still want to allow the explicit
	// queries to modify the resulting versions.
	var tentative []module.Version
	for _, cs := range upgrades {
		if cs.err != nil {
			base.Error(cs.err)
			continue
		}

		filtered, _, m, unique := r.disambiguate(cs)
		if !unique {
			_, m = r.chooseArbitrarily(filtered)
		}
		if m.Path == "" {
			// There is no viable candidate for the missing package.
			// Leave it unresolved.
			continue
		}
		tentative = append(tentative, m)
	}
	base.ExitIfErrors()

	changed = r.updateBuildList(ctx, tentative)
	return changed
}

// disambiguate eliminates candidates from cs that conflict with other module
// versions that have already been resolved. If there is only one (unique)
// remaining candidate, disambiguate returns that candidate, along with
// an indication of whether that result interprets cs.path as a package
//
// Note: we're only doing very simple disambiguation here. The goal is to
// reproduce the user's intent, not to find a solution that a human couldn't.
// In the vast majority of cases, we expect only one module per pathSet,
// but we want to give some minimal additional tools so that users can add an
// extra argument or two on the command line to resolve simple ambiguities.
func (r *resolver) disambiguate(cs pathSet) (filtered pathSet, isPackage bool, m module.Version, unique bool) {
	if len(cs.pkgMods) == 0 && cs.mod.Path == "" {
		panic("internal error: resolveIfUnambiguous called with empty pathSet")
	}

	for _, m := range cs.pkgMods {
		if _, ok := r.noneForPath(m.Path); ok {
			// A query with version "none" forces the candidate module to version
			// "none", so we cannot use any other version for that module.
			continue
		}

		if modload.MainModules.Contains(m.Path) {
			if m.Version == "" {
				return pathSet{}, true, m, true
			}
			// A main module can only be set to its own version.
			continue
		}

		vr, ok := r.resolvedVersion[m.Path]
		if !ok {
			// m is a viable answer to the query, but other answers may also
			// still be viable.
			filtered.pkgMods = append(filtered.pkgMods, m)
			continue
		}

		if vr.version != m.Version {
			// Some query forces the candidate module to a version other than this
			// one.
			//
			// The command could be something like
			//
			// 	go get example.com/foo/bar@none example.com/foo/bar/baz@latest
			//
			// in which case we *cannot* resolve the package from
			// example.com/foo/bar (because it is constrained to version
			// "none") and must fall through to module example.com/foo@latest.
			continue
		}

		// Some query forces the candidate module *to* the candidate version.
		// As a result, this candidate is the only viable choice to provide
		// its package(s): any other choice would result in an ambiguous import
		// for this path.
		//
		// For example, consider the command
		//
		// 	go get example.com/foo@latest example.com/foo/bar/baz@latest
		//
		// If modules example.com/foo and example.com/foo/bar both provide
		// package example.com/foo/bar/baz, then we *must* resolve the package
		// from example.com/foo: if we instead resolved it from
		// example.com/foo/bar, we would have two copies of the package.
		return pathSet{}, true, m, true
	}

	if cs.mod.Path != "" {
		vr, ok := r.resolvedVersion[cs.mod.Path]
		if !ok || vr.version == cs.mod.Version {
			filtered.mod = cs.mod
		}
	}

	if len(filtered.pkgMods) == 1 &&
		(filtered.mod.Path == "" || filtered.mod == filtered.pkgMods[0]) {
		// Exactly one viable module contains the package with the given path
		// (by far the common case), so we can resolve it unambiguously.
		return pathSet{}, true, filtered.pkgMods[0], true
	}

	if len(filtered.pkgMods) == 0 {
		// All modules that could provide the path as a package conflict with other
		// resolved arguments. If it can refer to a module instead, return that;
		// otherwise, this pathSet cannot be resolved (and we will return the
		// zero module.Version).
		return pathSet{}, false, filtered.mod, true
	}

	// The query remains ambiguous: there are at least two different modules
	// to which cs.path could refer.
	return filtered, false, module.Version{}, false
}

// chooseArbitrarily returns an arbitrary (but deterministic) module version
// from among those in the given set.
//
// chooseArbitrarily prefers module paths that were already in the build list at
// the start of 'go get', prefers modules that provide packages over those that
// do not, and chooses the first module meeting those criteria (so biases toward
// longer paths).
func (r *resolver) chooseArbitrarily(cs pathSet) (isPackage bool, m module.Version) {
	// Prefer to upgrade some module that was already in the build list.
	for _, m := range cs.pkgMods {
		if r.initialSelected(m.Path) != "none" {
			return true, m
		}
	}

	// Otherwise, arbitrarily choose the first module that provides the package.
	if len(cs.pkgMods) > 0 {
		return true, cs.pkgMods[0]
	}

	return false, cs.mod
}

// checkPackageProblems reloads packages for the given patterns and reports
// missing and ambiguous package errors. It also reports retractions and
// deprecations for resolved modules and modules needed to build named packages.
// It also adds a sum for each updated module in the build list if we had one
// before and didn't get one while loading packages.
//
// We skip missing-package errors earlier in the process, since we want to
// resolve pathSets ourselves, but at that point, we don't have enough context
// to log the package-import chains leading to each error.
func (r *resolver) checkPackageProblems(ctx context.Context, pkgPatterns []string) {
	defer base.ExitIfErrors()

	// Gather information about modules we might want to load retractions and
	// deprecations for. Loading this metadata requires at least one version
	// lookup per module, and we don't want to load information that's neither
	// relevant nor actionable.
	type modFlags int
	const (
		resolved modFlags = 1 << iota // version resolved by 'go get'
		named                         // explicitly named on command line or provides a named package
		hasPkg                        // needed to build named packages
		direct                        // provides a direct dependency of the main module
	)
	relevantMods := make(map[module.Version]modFlags)
	for path, reason := range r.resolvedVersion {
		m := module.Version{Path: path, Version: reason.version}
		relevantMods[m] |= resolved
	}

	// Reload packages, reporting errors for missing and ambiguous imports.
	if len(pkgPatterns) > 0 {
		// LoadPackages will print errors (since it has more context) but will not
		// exit, since we need to load retractions later.
		pkgOpts := modload.PackageOpts{
			VendorModulesInGOROOTSrc: true,
			LoadTests:                *getT,
			ResolveMissingImports:    false,
			AllowErrors:              true,
			SilenceNoGoErrors:        true,
		}
		matches, pkgs := modload.LoadPackages(ctx, pkgOpts, pkgPatterns...)
		for _, m := range matches {
			if len(m.Errs) > 0 {
				base.SetExitStatus(1)
				break
			}
		}
		for _, pkg := range pkgs {
			if dir, _, err := modload.Lookup("", false, pkg); err != nil {
				if dir != "" && errors.Is(err, imports.ErrNoGo) {
					// Since dir is non-empty, we must have located source files
					// associated with either the package or its test — ErrNoGo must
					// indicate that none of those source files happen to apply in this
					// configuration. If we are actually building the package (no -d
					// flag), we will report the problem then; otherwise, assume that the
					// user is going to build or test this package in some other
					// configuration and suppress the error.
					continue
				}

				base.SetExitStatus(1)
				if ambiguousErr := (*modload.AmbiguousImportError)(nil); errors.As(err, &ambiguousErr) {
					for _, m := range ambiguousErr.Modules {
						relevantMods[m] |= hasPkg
					}
				}
			}
			if m := modload.PackageModule(pkg); m.Path != "" {
				relevantMods[m] |= hasPkg
			}
		}
		for _, match := range matches {
			for _, pkg := range match.Pkgs {
				m := modload.PackageModule(pkg)
				relevantMods[m] |= named
			}
		}
	}

	reqs := modload.LoadModFile(ctx)
	for m := range relevantMods {
		if reqs.IsDirect(m.Path) {
			relevantMods[m] |= direct
		}
	}

	// Load retractions for modules mentioned on the command line and modules
	// needed to build named packages. We care about retractions of indirect
	// dependencies, since we might be able to upgrade away from them.
	type modMessage struct {
		m       module.Version
		message string
	}
	retractions := make([]modMessage, 0, len(relevantMods))
	for m, flags := range relevantMods {
		if flags&(resolved|named|hasPkg) != 0 {
			retractions = append(retractions, modMessage{m: m})
		}
	}
	sort.Slice(retractions, func(i, j int) bool { return retractions[i].m.Path < retractions[j].m.Path })
	for i := range retractions {
		i := i
		r.work.Add(func() {
			err := modload.CheckRetractions(ctx, retractions[i].m)
			if retractErr := (*modload.ModuleRetractedError)(nil); errors.As(err, &retractErr) {
				retractions[i].message = err.Error()
			}
		})
	}

	// Load deprecations for modules mentioned on the command line. Only load
	// deprecations for indirect dependencies if they're also direct dependencies
	// of the main module. Deprecations of purely indirect dependencies are
	// not actionable.
	deprecations := make([]modMessage, 0, len(relevantMods))
	for m, flags := range relevantMods {
		if flags&(resolved|named) != 0 || flags&(hasPkg|direct) == hasPkg|direct {
			deprecations = append(deprecations, modMessage{m: m})
		}
	}
	sort.Slice(deprecations, func(i, j int) bool { return deprecations[i].m.Path < deprecations[j].m.Path })
	for i := range deprecations {
		i := i
		r.work.Add(func() {
			deprecation, err := modload.CheckDeprecation(ctx, deprecations[i].m)
			if err != nil || deprecation == "" {
				return
			}
			deprecations[i].message = modload.ShortMessage(deprecation, "")
		})
	}

	// Load sums for updated modules that had sums before. When we update a
	// module, we may update another module in the build list that provides a
	// package in 'all' that wasn't loaded as part of this 'go get' command.
	// If we don't add a sum for that module, builds may fail later.
	// Note that an incidentally updated package could still import packages
	// from unknown modules or from modules in the build list that we didn't
	// need previously. We can't handle that case without loading 'all'.
	sumErrs := make([]error, len(r.buildList))
	for i := range r.buildList {
		i := i
		m := r.buildList[i]
		mActual := m
		if mRepl := modload.Replacement(m); mRepl.Path != "" {
			mActual = mRepl
		}
		old := module.Version{Path: m.Path, Version: r.initialVersion[m.Path]}
		if old.Version == "" {
			continue
		}
		oldActual := old
		if oldRepl := modload.Replacement(old); oldRepl.Path != "" {
			oldActual = oldRepl
		}
		if mActual == oldActual || mActual.Version == "" || !modfetch.HaveSum(oldActual) {
			continue
		}
		r.work.Add(func() {
			if _, err := modfetch.DownloadZip(ctx, mActual); err != nil {
				verb := "upgraded"
				if gover.ModCompare(m.Path, m.Version, old.Version) < 0 {
					verb = "downgraded"
				}
				replaced := ""
				if mActual != m {
					replaced = fmt.Sprintf(" (replaced by %s)", mActual)
				}
				err = fmt.Errorf("%s %s %s => %s%s: error finding sum for %s: %v", verb, m.Path, old.Version, m.Version, replaced, mActual, err)
				sumErrs[i] = err
			}
		})
	}

	<-r.work.Idle()

	// Report deprecations, then retractions, then errors fetching sums.
	// Only errors fetching sums are hard errors.
	for _, mm := range deprecations {
		if mm.message != "" {
			fmt.Fprintf(os.Stderr, "go: module %s is deprecated: %s\n", mm.m.Path, mm.message)
		}
	}
	var retractPath string
	for _, mm := range retractions {
		if mm.message != "" {
			fmt.Fprintf(os.Stderr, "go: warning: %v\n", mm.message)
			if retractPath == "" {
				retractPath = mm.m.Path
			} else {
				retractPath = "<module>"
			}
		}
	}
	if retractPath != "" {
		fmt.Fprintf(os.Stderr, "go: to switch to the latest unretracted version, run:\n\tgo get %s@latest\n", retractPath)
	}
	for _, err := range sumErrs {
		if err != nil {
			base.Error(err)
		}
	}
}

// reportChanges logs version changes to os.Stderr.
//
// reportChanges only logs changes to modules named on the command line and to
// explicitly required modules in go.mod. Most changes to indirect requirements
// are not relevant to the user and are not logged.
//
// reportChanges should be called after WriteGoMod.
func (r *resolver) reportChanges(oldReqs, newReqs []module.Version) {
	type change struct {
		path, old, new string
	}
	changes := make(map[string]change)

	// Collect changes in modules matched by command line arguments.
	for path, reason := range r.resolvedVersion {
		if gover.IsToolchain(path) {
			continue
		}
		old := r.initialVersion[path]
		new := reason.version
		if old != new && (old != "" || new != "none") {
			changes[path] = change{path, old, new}
		}
	}

	// Collect changes to explicit requirements in go.mod.
	for _, req := range oldReqs {
		if gover.IsToolchain(req.Path) {
			continue
		}
		path := req.Path
		old := req.Version
		new := r.buildListVersion[path]
		if old != new {
			changes[path] = change{path, old, new}
		}
	}
	for _, req := range newReqs {
		if gover.IsToolchain(req.Path) {
			continue
		}
		path := req.Path
		old := r.initialVersion[path]
		new := req.Version
		if old != new {
			changes[path] = change{path, old, new}
		}
	}

	// Toolchain diffs are easier than requirements: diff old and new directly.
	toolchainVersions := func(reqs []module.Version) (goV, toolchain string) {
		for _, req := range reqs {
			if req.Path == "go" {
				goV = req.Version
			}
			if req.Path == "toolchain" {
				toolchain = req.Version
			}
		}
		return
	}
	oldGo, oldToolchain := toolchainVersions(oldReqs)
	newGo, newToolchain := toolchainVersions(newReqs)
	if oldGo != newGo {
		changes["go"] = change{"go", oldGo, newGo}
	}
	if oldToolchain != newToolchain {
		changes["toolchain"] = change{"toolchain", oldToolchain, newToolchain}
	}

	sortedChanges := make([]change, 0, len(changes))
	for _, c := range changes {
		sortedChanges = append(sortedChanges, c)
	}
	sort.Slice(sortedChanges, func(i, j int) bool {
		pi := sortedChanges[i].path
		pj := sortedChanges[j].path
		if pi == pj {
			return false
		}
		// go first; toolchain second
		switch {
		case pi == "go":
			return true
		case pj == "go":
			return false
		case pi == "toolchain":
			return true
		case pj == "toolchain":
			return false
		}
		return pi < pj
	})

	for _, c := range sortedChanges {
		if c.old == "" {
			fmt.Fprintf(os.Stderr, "go: added %s %s\n", c.path, c.new)
		} else if c.new == "none" || c.new == "" {
			fmt.Fprintf(os.Stderr, "go: removed %s %s\n", c.path, c.old)
		} else if gover.ModCompare(c.path, c.new, c.old) > 0 {
			fmt.Fprintf(os.Stderr, "go: upgraded %s %s => %s\n", c.path, c.old, c.new)
			if c.path == "go" && gover.Compare(c.old, gover.ExplicitIndirectVersion) < 0 && gover.Compare(c.new, gover.ExplicitIndirectVersion) >= 0 {
				fmt.Fprintf(os.Stderr, "\tnote: expanded dependencies to upgrade to go %s or higher; run 'go mod tidy' to clean up\n", gover.ExplicitIndirectVersion)
			}

		} else {
			fmt.Fprintf(os.Stderr, "go: downgraded %s %s => %s\n", c.path, c.old, c.new)
		}
	}

	// TODO(golang.org/issue/33284): attribute changes to command line arguments.
	// For modules matched by command line arguments, this probably isn't
	// necessary, but it would be useful for unmatched direct dependencies of
	// the main module.
}

// resolve records that module m must be at its indicated version (which may be
// "none") due to query q. If some other query forces module m to be at a
// different version, resolve reports a conflict error.
func (r *resolver) resolve(q *query, m module.Version) {
	if m.Path == "" {
		panic("internal error: resolving a module.Version with an empty path")
	}

	if modload.MainModules.Contains(m.Path) && m.Version != "" {
		reportError(q, &modload.QueryMatchesMainModulesError{
			MainModules: []module.Version{{Path: m.Path}},
			Pattern:     q.pattern,
			Query:       q.version,
		})
		return
	}

	vr, ok := r.resolvedVersion[m.Path]
	if ok && vr.version != m.Version {
		reportConflict(q, m, vr)
		return
	}
	r.resolvedVersion[m.Path] = versionReason{m.Version, q}
	q.resolved = append(q.resolved, m)
}

// updateBuildList updates the module loader's global build list to be
// consistent with r.resolvedVersion, and to include additional modules
// provided that they do not conflict with the resolved versions.
//
// If the additional modules conflict with the resolved versions, they will be
// downgraded to a non-conflicting version (possibly "none").
//
// If the resulting build list is the same as the one resulting from the last
// call to updateBuildList, updateBuildList returns with changed=false.
func (r *resolver) updateBuildList(ctx context.Context, additions []module.Version) (changed bool) {
	defer base.ExitIfErrors()

	resolved := make([]module.Version, 0, len(r.resolvedVersion))
	for mPath, rv := range r.resolvedVersion {
		if !modload.MainModules.Contains(mPath) {
			resolved = append(resolved, module.Version{Path: mPath, Version: rv.version})
		}
	}

	changed, err := modload.EditBuildList(ctx, additions, resolved)
	if err != nil {
		if errors.Is(err, gover.ErrTooNew) {
			toolchain.SwitchOrFatal(ctx, err)
		}

		var constraint *modload.ConstraintError
		if !errors.As(err, &constraint) {
			base.Fatal(err)
		}

		if cfg.BuildV {
			// Log complete paths for the conflicts before we summarize them.
			for _, c := range constraint.Conflicts {
				fmt.Fprintf(os.Stderr, "go: %v\n", c.String())
			}
		}

		// modload.EditBuildList reports constraint errors at
		// the module level, but 'go get' operates on packages.
		// Rewrite the errors to explain them in terms of packages.
		reason := func(m module.Version) string {
			rv, ok := r.resolvedVersion[m.Path]
			if !ok {
				return fmt.Sprintf("(INTERNAL ERROR: no reason found for %v)", m)
			}
			return rv.reason.ResolvedString(module.Version{Path: m.Path, Version: rv.version})
		}
		for _, c := range constraint.Conflicts {
			adverb := ""
			if len(c.Path) > 2 {
				adverb = "indirectly "
			}
			firstReason := reason(c.Path[0])
			last := c.Path[len(c.Path)-1]
			if c.Err != nil {
				base.Errorf("go: %v %srequires %v: %v", firstReason, adverb, last, c.UnwrapModuleError())
			} else {
				base.Errorf("go: %v %srequires %v, not %v", firstReason, adverb, last, reason(c.Constraint))
			}
		}
		return false
	}
	if !changed {
		return false
	}

	mg, err := modload.LoadModGraph(ctx, "")
	if err != nil {
		toolchain.SwitchOrFatal(ctx, err)
	}

	r.buildList = mg.BuildList()
	r.buildListVersion = make(map[string]string, len(r.buildList))
	for _, m := range r.buildList {
		r.buildListVersion[m.Path] = m.Version
	}
	return true
}

func reqsFromGoMod(f *modfile.File) []module.Version {
	reqs := make([]module.Version, len(f.Require), 2+len(f.Require))
	for i, r := range f.Require {
		reqs[i] = r.Mod
	}
	if f.Go != nil {
		reqs = append(reqs, module.Version{Path: "go", Version: f.Go.Version})
	}
	if f.Toolchain != nil {
		reqs = append(reqs, module.Version{Path: "toolchain", Version: f.Toolchain.Name})
	}
	return reqs
}

// isNoSuchModuleVersion reports whether err indicates that the requested module
// does not exist at the requested version, either because the module does not
// exist at all or because it does not include that specific version.
func isNoSuchModuleVersion(err error) bool {
	var noMatch *modload.NoMatchingVersionError
	return errors.Is(err, os.ErrNotExist) || errors.As(err, &noMatch)
}

// isNoSuchPackageVersion reports whether err indicates that the requested
// package does not exist at the requested version, either because no module
// that could contain it exists at that version, or because every such module
// that does exist does not actually contain the package.
func isNoSuchPackageVersion(err error) bool {
	var noPackage *modload.PackageNotInModuleError
	return isNoSuchModuleVersion(err) || errors.As(err, &noPackage)
}
