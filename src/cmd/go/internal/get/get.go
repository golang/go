// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package get implements the ``go get'' command.
package get

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
	"cmd/go/internal/vcs"
	"cmd/go/internal/web"
	"cmd/go/internal/work"

	"golang.org/x/mod/module"
)

var CmdGet = &base.Command{
	UsageLine: "go get [-d] [-f] [-t] [-u] [-v] [-fix] [-insecure] [build flags] [packages]",
	Short:     "download and install packages and dependencies",
	Long: `
Get downloads the packages named by the import paths, along with their
dependencies. It then installs the named packages, like 'go install'.

The -d flag instructs get to stop after downloading the packages; that is,
it instructs get not to install the packages.

The -f flag, valid only when -u is set, forces get -u not to verify that
each package has been checked out from the source control repository
implied by its import path. This can be useful if the source is a local fork
of the original.

The -fix flag instructs get to run the fix tool on the downloaded packages
before resolving dependencies or building the code.

The -insecure flag permits fetching from repositories and resolving
custom domains using insecure schemes such as HTTP. Use with caution.
This flag is deprecated and will be removed in a future version of go.
The GOINSECURE environment variable should be used instead, since it
provides control over which packages may be retrieved using an insecure
scheme. See 'go help environment' for details.

The -t flag instructs get to also download the packages required to build
the tests for the specified packages.

The -u flag instructs get to use the network to update the named packages
and their dependencies. By default, get uses the network to check out
missing packages but does not use it to look for updates to existing packages.

The -v flag enables verbose progress and debug output.

Get also accepts build flags to control the installation. See 'go help build'.

When checking out a new package, get creates the target directory
GOPATH/src/<import-path>. If the GOPATH contains multiple entries,
get uses the first one. For more details see: 'go help gopath'.

When checking out or updating a package, get looks for a branch or tag
that matches the locally installed version of Go. The most important
rule is that if the local installation is running version "go1", get
searches for a branch or tag named "go1". If no such version exists
it retrieves the default branch of the package.

When go get checks out or updates a Git repository,
it also updates any git submodules referenced by the repository.

Get never checks out or updates code stored in vendor directories.

For more about specifying packages, see 'go help packages'.

For more about how 'go get' finds source code to
download, see 'go help importpath'.

This text describes the behavior of get when using GOPATH
to manage source code and dependencies.
If instead the go command is running in module-aware mode,
the details of get's flags and effects change, as does 'go help get'.
See 'go help modules' and 'go help module-get'.

See also: go build, go install, go clean.
	`,
}

var HelpGopathGet = &base.Command{
	UsageLine: "gopath-get",
	Short:     "legacy GOPATH go get",
	Long: `
The 'go get' command changes behavior depending on whether the
go command is running in module-aware mode or legacy GOPATH mode.
This help text, accessible as 'go help gopath-get' even in module-aware mode,
describes 'go get' as it operates in legacy GOPATH mode.

Usage: ` + CmdGet.UsageLine + `
` + CmdGet.Long,
}

var (
	getD   = CmdGet.Flag.Bool("d", false, "")
	getF   = CmdGet.Flag.Bool("f", false, "")
	getT   = CmdGet.Flag.Bool("t", false, "")
	getU   = CmdGet.Flag.Bool("u", false, "")
	getFix = CmdGet.Flag.Bool("fix", false, "")
)

func init() {
	work.AddBuildFlags(CmdGet, work.OmitModFlag|work.OmitModCommonFlags)
	CmdGet.Run = runGet // break init loop
	CmdGet.Flag.BoolVar(&cfg.Insecure, "insecure", cfg.Insecure, "")
}

func runGet(ctx context.Context, cmd *base.Command, args []string) {
	if cfg.ModulesEnabled {
		// Should not happen: main.go should install the separate module-enabled get code.
		base.Fatalf("go get: modules not implemented")
	}

	work.BuildInit()

	if *getF && !*getU {
		base.Fatalf("go get: cannot use -f flag without -u")
	}
	if cfg.Insecure {
		fmt.Fprintf(os.Stderr, "go get: -insecure flag is deprecated; see 'go help get' for details\n")
	}

	// Disable any prompting for passwords by Git.
	// Only has an effect for 2.3.0 or later, but avoiding
	// the prompt in earlier versions is just too hard.
	// If user has explicitly set GIT_TERMINAL_PROMPT=1, keep
	// prompting.
	// See golang.org/issue/9341 and golang.org/issue/12706.
	if os.Getenv("GIT_TERMINAL_PROMPT") == "" {
		os.Setenv("GIT_TERMINAL_PROMPT", "0")
	}

	// Disable any ssh connection pooling by Git.
	// If a Git subprocess forks a child into the background to cache a new connection,
	// that child keeps stdout/stderr open. After the Git subprocess exits,
	// os /exec expects to be able to read from the stdout/stderr pipe
	// until EOF to get all the data that the Git subprocess wrote before exiting.
	// The EOF doesn't come until the child exits too, because the child
	// is holding the write end of the pipe.
	// This is unfortunate, but it has come up at least twice
	// (see golang.org/issue/13453 and golang.org/issue/16104)
	// and confuses users when it does.
	// If the user has explicitly set GIT_SSH or GIT_SSH_COMMAND,
	// assume they know what they are doing and don't step on it.
	// But default to turning off ControlMaster.
	if os.Getenv("GIT_SSH") == "" && os.Getenv("GIT_SSH_COMMAND") == "" {
		os.Setenv("GIT_SSH_COMMAND", "ssh -o ControlMaster=no")
	}

	// Phase 1. Download/update.
	var stk load.ImportStack
	mode := 0
	if *getT {
		mode |= load.GetTestDeps
	}
	for _, pkg := range downloadPaths(args) {
		download(pkg, nil, &stk, mode)
	}
	base.ExitIfErrors()

	// Phase 2. Rescan packages and re-evaluate args list.

	// Code we downloaded and all code that depends on it
	// needs to be evicted from the package cache so that
	// the information will be recomputed. Instead of keeping
	// track of the reverse dependency information, evict
	// everything.
	load.ClearPackageCache()

	pkgs := load.PackagesAndErrors(ctx, args)
	load.CheckPackageErrors(pkgs)

	// Phase 3. Install.
	if *getD {
		// Download only.
		// Check delayed until now so that downloadPaths
		// and CheckPackageErrors have a chance to print errors.
		return
	}

	work.InstallPackages(ctx, args, pkgs)
}

// downloadPaths prepares the list of paths to pass to download.
// It expands ... patterns that can be expanded. If there is no match
// for a particular pattern, downloadPaths leaves it in the result list,
// in the hope that we can figure out the repository from the
// initial ...-free prefix.
func downloadPaths(patterns []string) []string {
	for _, arg := range patterns {
		if strings.Contains(arg, "@") {
			base.Fatalf("go: can only use path@version syntax with 'go get' and 'go install' in module-aware mode")
			continue
		}

		// Guard against 'go get x.go', a common mistake.
		// Note that package and module paths may end with '.go', so only print an error
		// if the argument has no slash or refers to an existing file.
		if strings.HasSuffix(arg, ".go") {
			if !strings.Contains(arg, "/") {
				base.Errorf("go get %s: arguments must be package or module paths", arg)
				continue
			}
			if fi, err := os.Stat(arg); err == nil && !fi.IsDir() {
				base.Errorf("go get: %s exists as a file, but 'go get' requires package arguments", arg)
			}
		}
	}
	base.ExitIfErrors()

	var pkgs []string
	for _, m := range search.ImportPathsQuiet(patterns) {
		if len(m.Pkgs) == 0 && strings.Contains(m.Pattern(), "...") {
			pkgs = append(pkgs, m.Pattern())
		} else {
			pkgs = append(pkgs, m.Pkgs...)
		}
	}
	return pkgs
}

// downloadCache records the import paths we have already
// considered during the download, to avoid duplicate work when
// there is more than one dependency sequence leading to
// a particular package.
var downloadCache = map[string]bool{}

// downloadRootCache records the version control repository
// root directories we have already considered during the download.
// For example, all the packages in the github.com/google/codesearch repo
// share the same root (the directory for that path), and we only need
// to run the hg commands to consider each repository once.
var downloadRootCache = map[string]bool{}

// download runs the download half of the get command
// for the package or pattern named by the argument.
func download(arg string, parent *load.Package, stk *load.ImportStack, mode int) {
	if mode&load.ResolveImport != 0 {
		// Caller is responsible for expanding vendor paths.
		panic("internal error: download mode has useVendor set")
	}
	load1 := func(path string, mode int) *load.Package {
		if parent == nil {
			mode := 0 // don't do module or vendor resolution
			return load.LoadImport(context.TODO(), path, base.Cwd, nil, stk, nil, mode)
		}
		return load.LoadImport(context.TODO(), path, parent.Dir, parent, stk, nil, mode|load.ResolveModule)
	}

	p := load1(arg, mode)
	if p.Error != nil && p.Error.Hard {
		base.Errorf("%s", p.Error)
		return
	}

	// loadPackage inferred the canonical ImportPath from arg.
	// Use that in the following to prevent hysteresis effects
	// in e.g. downloadCache and packageCache.
	// This allows invocations such as:
	//   mkdir -p $GOPATH/src/github.com/user
	//   cd $GOPATH/src/github.com/user
	//   go get ./foo
	// see: golang.org/issue/9767
	arg = p.ImportPath

	// There's nothing to do if this is a package in the standard library.
	if p.Standard {
		return
	}

	// Only process each package once.
	// (Unless we're fetching test dependencies for this package,
	// in which case we want to process it again.)
	if downloadCache[arg] && mode&load.GetTestDeps == 0 {
		return
	}
	downloadCache[arg] = true

	pkgs := []*load.Package{p}
	wildcardOkay := len(*stk) == 0
	isWildcard := false

	// Download if the package is missing, or update if we're using -u.
	if p.Dir == "" || *getU {
		// The actual download.
		stk.Push(arg)
		err := downloadPackage(p)
		if err != nil {
			base.Errorf("%s", &load.PackageError{ImportStack: stk.Copy(), Err: err})
			stk.Pop()
			return
		}
		stk.Pop()

		args := []string{arg}
		// If the argument has a wildcard in it, re-evaluate the wildcard.
		// We delay this until after reloadPackage so that the old entry
		// for p has been replaced in the package cache.
		if wildcardOkay && strings.Contains(arg, "...") {
			match := search.NewMatch(arg)
			if match.IsLocal() {
				match.MatchDirs()
				args = match.Dirs
			} else {
				match.MatchPackages()
				args = match.Pkgs
			}
			for _, err := range match.Errs {
				base.Errorf("%s", err)
			}
			isWildcard = true
		}

		// Clear all relevant package cache entries before
		// doing any new loads.
		load.ClearPackageCachePartial(args)

		pkgs = pkgs[:0]
		for _, arg := range args {
			// Note: load calls loadPackage or loadImport,
			// which push arg onto stk already.
			// Do not push here too, or else stk will say arg imports arg.
			p := load1(arg, mode)
			if p.Error != nil {
				base.Errorf("%s", p.Error)
				continue
			}
			pkgs = append(pkgs, p)
		}
	}

	// Process package, which might now be multiple packages
	// due to wildcard expansion.
	for _, p := range pkgs {
		if *getFix {
			files := base.RelPaths(p.InternalAllGoFiles())
			base.Run(cfg.BuildToolexec, str.StringList(base.Tool("fix"), files))

			// The imports might have changed, so reload again.
			p = load.ReloadPackageNoFlags(arg, stk)
			if p.Error != nil {
				base.Errorf("%s", p.Error)
				return
			}
		}

		if isWildcard {
			// Report both the real package and the
			// wildcard in any error message.
			stk.Push(p.ImportPath)
		}

		// Process dependencies, now that we know what they are.
		imports := p.Imports
		if mode&load.GetTestDeps != 0 {
			// Process test dependencies when -t is specified.
			// (But don't get test dependencies for test dependencies:
			// we always pass mode 0 to the recursive calls below.)
			imports = str.StringList(imports, p.TestImports, p.XTestImports)
		}
		for i, path := range imports {
			if path == "C" {
				continue
			}
			// Fail fast on import naming full vendor path.
			// Otherwise expand path as needed for test imports.
			// Note that p.Imports can have additional entries beyond p.Internal.Build.Imports.
			orig := path
			if i < len(p.Internal.Build.Imports) {
				orig = p.Internal.Build.Imports[i]
			}
			if j, ok := load.FindVendor(orig); ok {
				stk.Push(path)
				err := &load.PackageError{
					ImportStack: stk.Copy(),
					Err:         load.ImportErrorf(path, "%s must be imported as %s", path, path[j+len("vendor/"):]),
				}
				stk.Pop()
				base.Errorf("%s", err)
				continue
			}
			// If this is a test import, apply module and vendor lookup now.
			// We cannot pass ResolveImport to download, because
			// download does caching based on the value of path,
			// so it must be the fully qualified path already.
			if i >= len(p.Imports) {
				path = load.ResolveImportPath(p, path)
			}
			download(path, p, stk, 0)
		}

		if isWildcard {
			stk.Pop()
		}
	}
}

// downloadPackage runs the create or download command
// to make the first copy of or update a copy of the given package.
func downloadPackage(p *load.Package) error {
	var (
		vcsCmd         *vcs.Cmd
		repo, rootPath string
		err            error
		blindRepo      bool // set if the repo has unusual configuration
	)

	// p can be either a real package, or a pseudo-package whose “import path” is
	// actually a wildcard pattern.
	// Trim the path at the element containing the first wildcard,
	// and hope that it applies to the wildcarded parts too.
	// This makes 'go get rsc.io/pdf/...' work in a fresh GOPATH.
	importPrefix := p.ImportPath
	if i := strings.Index(importPrefix, "..."); i >= 0 {
		slash := strings.LastIndexByte(importPrefix[:i], '/')
		if slash < 0 {
			return fmt.Errorf("cannot expand ... in %q", p.ImportPath)
		}
		importPrefix = importPrefix[:slash]
	}
	if err := checkImportPath(importPrefix); err != nil {
		return fmt.Errorf("%s: invalid import path: %v", p.ImportPath, err)
	}
	security := web.SecureOnly
	if cfg.Insecure || module.MatchPrefixPatterns(cfg.GOINSECURE, importPrefix) {
		security = web.Insecure
	}

	if p.Internal.Build.SrcRoot != "" {
		// Directory exists. Look for checkout along path to src.
		vcsCmd, rootPath, err = vcs.FromDir(p.Dir, p.Internal.Build.SrcRoot)
		if err != nil {
			return err
		}
		repo = "<local>" // should be unused; make distinctive

		// Double-check where it came from.
		if *getU && vcsCmd.RemoteRepo != nil {
			dir := filepath.Join(p.Internal.Build.SrcRoot, filepath.FromSlash(rootPath))
			remote, err := vcsCmd.RemoteRepo(vcsCmd, dir)
			if err != nil {
				// Proceed anyway. The package is present; we likely just don't understand
				// the repo configuration (e.g. unusual remote protocol).
				blindRepo = true
			}
			repo = remote
			if !*getF && err == nil {
				if rr, err := vcs.RepoRootForImportPath(importPrefix, vcs.IgnoreMod, security); err == nil {
					repo := rr.Repo
					if rr.VCS.ResolveRepo != nil {
						resolved, err := rr.VCS.ResolveRepo(rr.VCS, dir, repo)
						if err == nil {
							repo = resolved
						}
					}
					if remote != repo && rr.IsCustom {
						return fmt.Errorf("%s is a custom import path for %s, but %s is checked out from %s", rr.Root, repo, dir, remote)
					}
				}
			}
		}
	} else {
		// Analyze the import path to determine the version control system,
		// repository, and the import path for the root of the repository.
		rr, err := vcs.RepoRootForImportPath(importPrefix, vcs.IgnoreMod, security)
		if err != nil {
			return err
		}
		vcsCmd, repo, rootPath = rr.VCS, rr.Repo, rr.Root
	}
	if !blindRepo && !vcsCmd.IsSecure(repo) && security != web.Insecure {
		return fmt.Errorf("cannot download, %v uses insecure protocol", repo)
	}

	if p.Internal.Build.SrcRoot == "" {
		// Package not found. Put in first directory of $GOPATH.
		list := filepath.SplitList(cfg.BuildContext.GOPATH)
		if len(list) == 0 {
			return fmt.Errorf("cannot download, $GOPATH not set. For more details see: 'go help gopath'")
		}
		// Guard against people setting GOPATH=$GOROOT.
		if filepath.Clean(list[0]) == filepath.Clean(cfg.GOROOT) {
			return fmt.Errorf("cannot download, $GOPATH must not be set to $GOROOT. For more details see: 'go help gopath'")
		}
		if _, err := os.Stat(filepath.Join(list[0], "src/cmd/go/alldocs.go")); err == nil {
			return fmt.Errorf("cannot download, %s is a GOROOT, not a GOPATH. For more details see: 'go help gopath'", list[0])
		}
		p.Internal.Build.Root = list[0]
		p.Internal.Build.SrcRoot = filepath.Join(list[0], "src")
		p.Internal.Build.PkgRoot = filepath.Join(list[0], "pkg")
	}
	root := filepath.Join(p.Internal.Build.SrcRoot, filepath.FromSlash(rootPath))

	if err := vcs.CheckNested(vcsCmd, root, p.Internal.Build.SrcRoot); err != nil {
		return err
	}

	// If we've considered this repository already, don't do it again.
	if downloadRootCache[root] {
		return nil
	}
	downloadRootCache[root] = true

	if cfg.BuildV {
		fmt.Fprintf(os.Stderr, "%s (download)\n", rootPath)
	}

	// Check that this is an appropriate place for the repo to be checked out.
	// The target directory must either not exist or have a repo checked out already.
	meta := filepath.Join(root, "."+vcsCmd.Cmd)
	if _, err := os.Stat(meta); err != nil {
		// Metadata file or directory does not exist. Prepare to checkout new copy.
		// Some version control tools require the target directory not to exist.
		// We require that too, just to avoid stepping on existing work.
		if _, err := os.Stat(root); err == nil {
			return fmt.Errorf("%s exists but %s does not - stale checkout?", root, meta)
		}

		_, err := os.Stat(p.Internal.Build.Root)
		gopathExisted := err == nil

		// Some version control tools require the parent of the target to exist.
		parent, _ := filepath.Split(root)
		if err = os.MkdirAll(parent, 0777); err != nil {
			return err
		}
		if cfg.BuildV && !gopathExisted && p.Internal.Build.Root == cfg.BuildContext.GOPATH {
			fmt.Fprintf(os.Stderr, "created GOPATH=%s; see 'go help gopath'\n", p.Internal.Build.Root)
		}

		if err = vcsCmd.Create(root, repo); err != nil {
			return err
		}
	} else {
		// Metadata directory does exist; download incremental updates.
		if err = vcsCmd.Download(root); err != nil {
			return err
		}
	}

	if cfg.BuildN {
		// Do not show tag sync in -n; it's noise more than anything,
		// and since we're not running commands, no tag will be found.
		// But avoid printing nothing.
		fmt.Fprintf(os.Stderr, "# cd %s; %s sync/update\n", root, vcsCmd.Cmd)
		return nil
	}

	// Select and sync to appropriate version of the repository.
	tags, err := vcsCmd.Tags(root)
	if err != nil {
		return err
	}
	vers := runtime.Version()
	if i := strings.Index(vers, " "); i >= 0 {
		vers = vers[:i]
	}
	if err := vcsCmd.TagSync(root, selectTag(vers, tags)); err != nil {
		return err
	}

	return nil
}

// selectTag returns the closest matching tag for a given version.
// Closest means the latest one that is not after the current release.
// Version "goX" (or "goX.Y" or "goX.Y.Z") matches tags of the same form.
// Version "release.rN" matches tags of the form "go.rN" (N being a floating-point number).
// Version "weekly.YYYY-MM-DD" matches tags like "go.weekly.YYYY-MM-DD".
//
// NOTE(rsc): Eventually we will need to decide on some logic here.
// For now, there is only "go1". This matches the docs in go help get.
func selectTag(goVersion string, tags []string) (match string) {
	for _, t := range tags {
		if t == "go1" {
			return "go1"
		}
	}
	return ""
}

// checkImportPath is like module.CheckImportPath, but it forbids leading dots
// in path elements. This can lead to 'go get' creating .git and other VCS
// directories in places we might run VCS tools later.
func checkImportPath(path string) error {
	if err := module.CheckImportPath(path); err != nil {
		return err
	}
	checkElem := func(elem string) error {
		if elem[0] == '.' {
			return fmt.Errorf("malformed import path %q: leading dot in path element", path)
		}
		return nil
	}
	elemStart := 0
	for i, r := range path {
		if r == '/' {
			if err := checkElem(path[elemStart:]); err != nil {
				return err
			}
			elemStart = i + 1
		}
	}
	if err := checkElem(path[elemStart:]); err != nil {
		return err
	}
	return nil
}
