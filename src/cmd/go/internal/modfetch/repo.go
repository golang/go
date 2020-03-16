// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"time"

	"cmd/go/internal/cfg"
	"cmd/go/internal/get"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/par"
	"cmd/go/internal/str"
	web "cmd/go/internal/web"

	"golang.org/x/mod/semver"
)

const traceRepo = false // trace all repo actions, for debugging

// A Repo represents a repository storing all versions of a single module.
// It must be safe for simultaneous use by multiple goroutines.
type Repo interface {
	// ModulePath returns the module path.
	ModulePath() string

	// Versions lists all known versions with the given prefix.
	// Pseudo-versions are not included.
	// Versions should be returned sorted in semver order
	// (implementations can use SortVersions).
	Versions(prefix string) ([]string, error)

	// Stat returns information about the revision rev.
	// A revision can be any identifier known to the underlying service:
	// commit hash, branch, tag, and so on.
	Stat(rev string) (*RevInfo, error)

	// Latest returns the latest revision on the default branch,
	// whatever that means in the underlying source code repository.
	// It is only used when there are no tagged versions.
	Latest() (*RevInfo, error)

	// GoMod returns the go.mod file for the given version.
	GoMod(version string) (data []byte, err error)

	// Zip writes a zip file for the given version to dst.
	Zip(dst io.Writer, version string) error
}

// A Rev describes a single revision in a module repository.
type RevInfo struct {
	Version string    // suggested version string for this revision
	Time    time.Time // commit time

	// These fields are used for Stat of arbitrary rev,
	// but they are not recorded when talking about module versions.
	Name  string `json:"-"` // complete ID in underlying repository
	Short string `json:"-"` // shortened ID, for use in pseudo-version
}

// Re: module paths, import paths, repository roots, and lookups
//
// A module is a collection of Go packages stored in a file tree
// with a go.mod file at the root of the tree.
// The go.mod defines the module path, which is the import path
// corresponding to the root of the file tree.
// The import path of a directory within that file tree is the module path
// joined with the name of the subdirectory relative to the root.
//
// For example, the module with path rsc.io/qr corresponds to the
// file tree in the repository https://github.com/rsc/qr.
// That file tree has a go.mod that says "module rsc.io/qr".
// The package in the root directory has import path "rsc.io/qr".
// The package in the gf256 subdirectory has import path "rsc.io/qr/gf256".
// In this example, "rsc.io/qr" is both a module path and an import path.
// But "rsc.io/qr/gf256" is only an import path, not a module path:
// it names an importable package, but not a module.
//
// As a special case to incorporate code written before modules were
// introduced, if a path p resolves using the pre-module "go get" lookup
// to the root of a source code repository without a go.mod file,
// that repository is treated as if it had a go.mod in its root directory
// declaring module path p. (The go.mod is further considered to
// contain requirements corresponding to any legacy version
// tracking format such as Gopkg.lock, vendor/vendor.conf, and so on.)
//
// The presentation so far ignores the fact that a source code repository
// has many different versions of a file tree, and those versions may
// differ in whether a particular go.mod exists and what it contains.
// In fact there is a well-defined mapping only from a module path, version
// pair - often written path@version - to a particular file tree.
// For example rsc.io/qr@v0.1.0 depends on the "implicit go.mod at root of
// repository" rule, while rsc.io/qr@v0.2.0 has an explicit go.mod.
// Because the "go get" import paths rsc.io/qr and github.com/rsc/qr
// both redirect to the Git repository https://github.com/rsc/qr,
// github.com/rsc/qr@v0.1.0 is the same file tree as rsc.io/qr@v0.1.0
// but a different module (a different name). In contrast, since v0.2.0
// of that repository has an explicit go.mod that declares path rsc.io/qr,
// github.com/rsc/qr@v0.2.0 is an invalid module path, version pair.
// Before modules, import comments would have had the same effect.
//
// The set of import paths associated with a given module path is
// clearly not fixed: at the least, new directories with new import paths
// can always be added. But another potential operation is to split a
// subtree out of a module into its own module. If done carefully,
// this operation can be done while preserving compatibility for clients.
// For example, suppose that we want to split rsc.io/qr/gf256 into its
// own module, so that there would be two modules rsc.io/qr and rsc.io/qr/gf256.
// Then we can simultaneously issue rsc.io/qr v0.3.0 (dropping the gf256 subdirectory)
// and rsc.io/qr/gf256 v0.1.0, including in their respective go.mod
// cyclic requirements pointing at each other: rsc.io/qr v0.3.0 requires
// rsc.io/qr/gf256 v0.1.0 and vice versa. Then a build can be
// using an older rsc.io/qr module that includes the gf256 package, but if
// it adds a requirement on either the newer rsc.io/qr or the newer
// rsc.io/qr/gf256 module, it will automatically add the requirement
// on the complementary half, ensuring both that rsc.io/qr/gf256 is
// available for importing by the build and also that it is only defined
// by a single module. The gf256 package could move back into the
// original by another simultaneous release of rsc.io/qr v0.4.0 including
// the gf256 subdirectory and an rsc.io/qr/gf256 v0.2.0 with no code
// in its root directory, along with a new requirement cycle.
// The ability to shift module boundaries in this way is expected to be
// important in large-scale program refactorings, similar to the ones
// described in https://talks.golang.org/2016/refactor.article.
//
// The possibility of shifting module boundaries reemphasizes
// that you must know both the module path and its version
// to determine the set of packages provided directly by that module.
//
// On top of all this, it is possible for a single code repository
// to contain multiple modules, either in branches or subdirectories,
// as a limited kind of monorepo. For example rsc.io/qr/v2,
// the v2.x.x continuation of rsc.io/qr, is expected to be found
// in v2-tagged commits in https://github.com/rsc/qr, either
// in the root or in a v2 subdirectory, disambiguated by go.mod.
// Again the precise file tree corresponding to a module
// depends on which version we are considering.
//
// It is also possible for the underlying repository to change over time,
// without changing the module path. If I copy the github repo over
// to https://bitbucket.org/rsc/qr and update https://rsc.io/qr?go-get=1,
// then clients of all versions should start fetching from bitbucket
// instead of github. That is, in contrast to the exact file tree,
// the location of the source code repository associated with a module path
// does not depend on the module version. (This is by design, as the whole
// point of these redirects is to allow package authors to establish a stable
// name that can be updated as code moves from one service to another.)
//
// All of this is important background for the lookup APIs defined in this
// file.
//
// The Lookup function takes a module path and returns a Repo representing
// that module path. Lookup can do only a little with the path alone.
// It can check that the path is well-formed (see semver.CheckPath)
// and it can check that the path can be resolved to a target repository.
// To avoid version control access except when absolutely necessary,
// Lookup does not attempt to connect to the repository itself.
//
// The ImportRepoRev function is a variant of Import which is limited
// to code in a source code repository at a particular revision identifier
// (usually a commit hash or source code repository tag, not necessarily
// a module version).
// ImportRepoRev is used when converting legacy dependency requirements
// from older systems into go.mod files. Those older systems worked
// at either package or repository granularity, and most of the time they
// recorded commit hashes, not tagged versions.

var lookupCache par.Cache

type lookupCacheKey struct {
	proxy, path string
}

// Lookup returns the module with the given module path,
// fetched through the given proxy.
//
// The distinguished proxy "direct" indicates that the path should be fetched
// from its origin, and "noproxy" indicates that the patch should be fetched
// directly only if GONOPROXY matches the given path.
//
// For the distinguished proxy "off", Lookup always returns a non-nil error.
//
// A successful return does not guarantee that the module
// has any defined versions.
func Lookup(proxy, path string) (Repo, error) {
	if traceRepo {
		defer logCall("Lookup(%q, %q)", proxy, path)()
	}

	type cached struct {
		r   Repo
		err error
	}
	c := lookupCache.Do(lookupCacheKey{proxy, path}, func() interface{} {
		r, err := lookup(proxy, path)
		if err == nil {
			if traceRepo {
				r = newLoggingRepo(r)
			}
			r = newCachingRepo(r)
		}
		return cached{r, err}
	}).(cached)

	return c.r, c.err
}

// lookup returns the module with the given module path.
func lookup(proxy, path string) (r Repo, err error) {
	if cfg.BuildMod == "vendor" {
		return nil, errLookupDisabled
	}

	if str.GlobsMatchPath(cfg.GONOPROXY, path) {
		switch proxy {
		case "noproxy", "direct":
			return lookupDirect(path)
		default:
			return nil, errNoproxy
		}
	}

	switch proxy {
	case "off":
		return nil, errProxyOff
	case "direct":
		return lookupDirect(path)
	case "noproxy":
		return nil, errUseProxy
	default:
		return newProxyRepo(proxy, path)
	}
}

type lookupDisabledError struct{}

func (lookupDisabledError) Error() string {
	if cfg.BuildModReason == "" {
		return fmt.Sprintf("module lookup disabled by -mod=%s", cfg.BuildMod)
	}
	return fmt.Sprintf("module lookup disabled by -mod=%s\n\t(%s)", cfg.BuildMod, cfg.BuildModReason)
}

var errLookupDisabled error = lookupDisabledError{}

var (
	errProxyOff       = notExistErrorf("module lookup disabled by GOPROXY=off")
	errNoproxy  error = notExistErrorf("disabled by GOPRIVATE/GONOPROXY")
	errUseProxy error = notExistErrorf("path does not match GOPRIVATE/GONOPROXY")
)

func lookupDirect(path string) (Repo, error) {
	security := web.SecureOnly

	if allowInsecure(path) {
		security = web.Insecure
	}
	rr, err := get.RepoRootForImportPath(path, get.PreferMod, security)
	if err != nil {
		// We don't know where to find code for a module with this path.
		return nil, notExistError{err: err}
	}

	if rr.VCS == "mod" {
		// Fetch module from proxy with base URL rr.Repo.
		return newProxyRepo(rr.Repo, path)
	}

	code, err := lookupCodeRepo(rr)
	if err != nil {
		return nil, err
	}
	return newCodeRepo(code, rr.Root, path)
}

func lookupCodeRepo(rr *get.RepoRoot) (codehost.Repo, error) {
	code, err := codehost.NewRepo(rr.VCS, rr.Repo)
	if err != nil {
		if _, ok := err.(*codehost.VCSError); ok {
			return nil, err
		}
		return nil, fmt.Errorf("lookup %s: %v", rr.Root, err)
	}
	return code, nil
}

// ImportRepoRev returns the module and version to use to access
// the given import path loaded from the source code repository that
// the original "go get" would have used, at the specific repository revision
// (typically a commit hash, but possibly also a source control tag).
func ImportRepoRev(path, rev string) (Repo, *RevInfo, error) {
	if cfg.BuildMod == "vendor" || cfg.BuildMod == "readonly" {
		return nil, nil, fmt.Errorf("repo version lookup disabled by -mod=%s", cfg.BuildMod)
	}

	// Note: Because we are converting a code reference from a legacy
	// version control system, we ignore meta tags about modules
	// and use only direct source control entries (get.IgnoreMod).
	security := web.SecureOnly
	if allowInsecure(path) {
		security = web.Insecure
	}
	rr, err := get.RepoRootForImportPath(path, get.IgnoreMod, security)
	if err != nil {
		return nil, nil, err
	}

	code, err := lookupCodeRepo(rr)
	if err != nil {
		return nil, nil, err
	}

	revInfo, err := code.Stat(rev)
	if err != nil {
		return nil, nil, err
	}

	// TODO: Look in repo to find path, check for go.mod files.
	// For now we're just assuming rr.Root is the module path,
	// which is true in the absence of go.mod files.

	repo, err := newCodeRepo(code, rr.Root, rr.Root)
	if err != nil {
		return nil, nil, err
	}

	info, err := repo.(*codeRepo).convert(revInfo, rev)
	if err != nil {
		return nil, nil, err
	}
	return repo, info, nil
}

func SortVersions(list []string) {
	sort.Slice(list, func(i, j int) bool {
		cmp := semver.Compare(list[i], list[j])
		if cmp != 0 {
			return cmp < 0
		}
		return list[i] < list[j]
	})
}

// A loggingRepo is a wrapper around an underlying Repo
// that prints a log message at the start and end of each call.
// It can be inserted when debugging.
type loggingRepo struct {
	r Repo
}

func newLoggingRepo(r Repo) *loggingRepo {
	return &loggingRepo{r}
}

// logCall prints a log message using format and args and then
// also returns a function that will print the same message again,
// along with the elapsed time.
// Typical usage is:
//
//	defer logCall("hello %s", arg)()
//
// Note the final ().
func logCall(format string, args ...interface{}) func() {
	start := time.Now()
	fmt.Fprintf(os.Stderr, "+++ %s\n", fmt.Sprintf(format, args...))
	return func() {
		fmt.Fprintf(os.Stderr, "%.3fs %s\n", time.Since(start).Seconds(), fmt.Sprintf(format, args...))
	}
}

func (l *loggingRepo) ModulePath() string {
	return l.r.ModulePath()
}

func (l *loggingRepo) Versions(prefix string) (tags []string, err error) {
	defer logCall("Repo[%s]: Versions(%q)", l.r.ModulePath(), prefix)()
	return l.r.Versions(prefix)
}

func (l *loggingRepo) Stat(rev string) (*RevInfo, error) {
	defer logCall("Repo[%s]: Stat(%q)", l.r.ModulePath(), rev)()
	return l.r.Stat(rev)
}

func (l *loggingRepo) Latest() (*RevInfo, error) {
	defer logCall("Repo[%s]: Latest()", l.r.ModulePath())()
	return l.r.Latest()
}

func (l *loggingRepo) GoMod(version string) ([]byte, error) {
	defer logCall("Repo[%s]: GoMod(%q)", l.r.ModulePath(), version)()
	return l.r.GoMod(version)
}

func (l *loggingRepo) Zip(dst io.Writer, version string) error {
	dstName := "_"
	if dst, ok := dst.(interface{ Name() string }); ok {
		dstName = strconv.Quote(dst.Name())
	}
	defer logCall("Repo[%s]: Zip(%s, %q)", l.r.ModulePath(), dstName, version)()
	return l.r.Zip(dst, version)
}

// A notExistError is like os.ErrNotExist, but with a custom message
type notExistError struct {
	err error
}

func notExistErrorf(format string, args ...interface{}) error {
	return notExistError{fmt.Errorf(format, args...)}
}

func (e notExistError) Error() string {
	return e.err.Error()
}

func (notExistError) Is(target error) bool {
	return target == os.ErrNotExist
}

func (e notExistError) Unwrap() error {
	return e.err
}
