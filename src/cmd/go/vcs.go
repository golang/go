// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// A vcsCmd describes how to use a version control system
// like Mercurial, Git, or Subversion.
type vcsCmd struct {
	name string
	cmd  string // name of binary to invoke command

	createCmd   string // command to download a fresh copy of a repository
	downloadCmd string // command to download updates into an existing repository

	tagCmd         []tagCmd // commands to list tags
	tagLookupCmd   []tagCmd // commands to lookup tags before running tagSyncCmd
	tagSyncCmd     string   // command to sync to specific tag
	tagSyncDefault string   // command to sync to default tag

	scheme  []string
	pingCmd string

	remoteRepo  func(v *vcsCmd, rootDir string) (remoteRepo string, err error)
	resolveRepo func(v *vcsCmd, rootDir, remoteRepo string) (realRepo string, err error)
}

// A tagCmd describes a command to list available tags
// that can be passed to tagSyncCmd.
type tagCmd struct {
	cmd     string // command to list tags
	pattern string // regexp to extract tags from list
}

// vcsList lists the known version control systems
var vcsList = []*vcsCmd{
	vcsHg,
	vcsGit,
	vcsSvn,
	vcsBzr,
}

// vcsByCmd returns the version control system for the given
// command name (hg, git, svn, bzr).
func vcsByCmd(cmd string) *vcsCmd {
	for _, vcs := range vcsList {
		if vcs.cmd == cmd {
			return vcs
		}
	}
	return nil
}

// vcsHg describes how to use Mercurial.
var vcsHg = &vcsCmd{
	name: "Mercurial",
	cmd:  "hg",

	createCmd:   "clone -U {repo} {dir}",
	downloadCmd: "pull",

	// We allow both tag and branch names as 'tags'
	// for selecting a version.  This lets people have
	// a go.release.r60 branch and a go1 branch
	// and make changes in both, without constantly
	// editing .hgtags.
	tagCmd: []tagCmd{
		{"tags", `^(\S+)`},
		{"branches", `^(\S+)`},
	},
	tagSyncCmd:     "update -r {tag}",
	tagSyncDefault: "update default",

	scheme:     []string{"https", "http", "ssh"},
	pingCmd:    "identify {scheme}://{repo}",
	remoteRepo: hgRemoteRepo,
}

func hgRemoteRepo(vcsHg *vcsCmd, rootDir string) (remoteRepo string, err error) {
	out, err := vcsHg.runOutput(rootDir, "paths default")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// vcsGit describes how to use Git.
var vcsGit = &vcsCmd{
	name: "Git",
	cmd:  "git",

	createCmd:   "clone {repo} {dir}",
	downloadCmd: "pull --ff-only",

	tagCmd: []tagCmd{
		// tags/xxx matches a git tag named xxx
		// origin/xxx matches a git branch named xxx on the default remote repository
		{"show-ref", `(?:tags|origin)/(\S+)$`},
	},
	tagLookupCmd: []tagCmd{
		{"show-ref tags/{tag} origin/{tag}", `((?:tags|origin)/\S+)$`},
	},
	tagSyncCmd:     "checkout {tag}",
	tagSyncDefault: "checkout master",

	scheme:     []string{"git", "https", "http", "git+ssh"},
	pingCmd:    "ls-remote {scheme}://{repo}",
	remoteRepo: gitRemoteRepo,
}

func gitRemoteRepo(vcsGit *vcsCmd, rootDir string) (remoteRepo string, err error) {
	outb, err := vcsGit.runOutput(rootDir, "remote -v")
	if err != nil {
		return "", err
	}
	out := string(outb)

	// Expect:
	// origin	https://github.com/rsc/pdf (fetch)
	// origin	https://github.com/rsc/pdf (push)
	// use first line only.

	if !strings.HasPrefix(out, "origin\t") {
		return "", fmt.Errorf("unable to parse output of git remote -v")
	}
	out = strings.TrimPrefix(out, "origin\t")
	i := strings.Index(out, "\n")
	if i < 0 {
		return "", fmt.Errorf("unable to parse output of git remote -v")
	}
	out = out[:i]
	i = strings.LastIndex(out, " ")
	if i < 0 {
		return "", fmt.Errorf("unable to parse output of git remote -v")
	}
	out = out[:i]
	return strings.TrimSpace(string(out)), nil
}

// vcsBzr describes how to use Bazaar.
var vcsBzr = &vcsCmd{
	name: "Bazaar",
	cmd:  "bzr",

	createCmd: "branch {repo} {dir}",

	// Without --overwrite bzr will not pull tags that changed.
	// Replace by --overwrite-tags after http://pad.lv/681792 goes in.
	downloadCmd: "pull --overwrite",

	tagCmd:         []tagCmd{{"tags", `^(\S+)`}},
	tagSyncCmd:     "update -r {tag}",
	tagSyncDefault: "update -r revno:-1",

	scheme:      []string{"https", "http", "bzr", "bzr+ssh"},
	pingCmd:     "info {scheme}://{repo}",
	remoteRepo:  bzrRemoteRepo,
	resolveRepo: bzrResolveRepo,
}

func bzrRemoteRepo(vcsBzr *vcsCmd, rootDir string) (remoteRepo string, err error) {
	outb, err := vcsBzr.runOutput(rootDir, "config parent_location")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(outb)), nil
}

func bzrResolveRepo(vcsBzr *vcsCmd, rootDir, remoteRepo string) (realRepo string, err error) {
	outb, err := vcsBzr.runOutput(rootDir, "info "+remoteRepo)
	if err != nil {
		return "", err
	}
	out := string(outb)

	// Expect:
	// ...
	//   (branch root|repository branch): <URL>
	// ...

	found := false
	for _, prefix := range []string{"\n  branch root: ", "\n  repository branch: "} {
		i := strings.Index(out, prefix)
		if i >= 0 {
			out = out[i+len(prefix):]
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("unable to parse output of bzr info")
	}

	i := strings.Index(out, "\n")
	if i < 0 {
		return "", fmt.Errorf("unable to parse output of bzr info")
	}
	out = out[:i]
	return strings.TrimSpace(string(out)), nil
}

// vcsSvn describes how to use Subversion.
var vcsSvn = &vcsCmd{
	name: "Subversion",
	cmd:  "svn",

	createCmd:   "checkout {repo} {dir}",
	downloadCmd: "update",

	// There is no tag command in subversion.
	// The branch information is all in the path names.

	scheme:     []string{"https", "http", "svn", "svn+ssh"},
	pingCmd:    "info {scheme}://{repo}",
	remoteRepo: svnRemoteRepo,
}

func svnRemoteRepo(vcsSvn *vcsCmd, rootDir string) (remoteRepo string, err error) {
	outb, err := vcsSvn.runOutput(rootDir, "info")
	if err != nil {
		return "", err
	}
	out := string(outb)

	// Expect:
	// ...
	// Repository Root: <URL>
	// ...

	i := strings.Index(out, "\nRepository Root: ")
	if i < 0 {
		return "", fmt.Errorf("unable to parse output of svn info")
	}
	out = out[i+len("\nRepository Root: "):]
	i = strings.Index(out, "\n")
	if i < 0 {
		return "", fmt.Errorf("unable to parse output of svn info")
	}
	out = out[:i]
	return strings.TrimSpace(string(out)), nil
}

func (v *vcsCmd) String() string {
	return v.name
}

// run runs the command line cmd in the given directory.
// keyval is a list of key, value pairs.  run expands
// instances of {key} in cmd into value, but only after
// splitting cmd into individual arguments.
// If an error occurs, run prints the command line and the
// command's combined stdout+stderr to standard error.
// Otherwise run discards the command's output.
func (v *vcsCmd) run(dir string, cmd string, keyval ...string) error {
	_, err := v.run1(dir, cmd, keyval, true)
	return err
}

// runVerboseOnly is like run but only generates error output to standard error in verbose mode.
func (v *vcsCmd) runVerboseOnly(dir string, cmd string, keyval ...string) error {
	_, err := v.run1(dir, cmd, keyval, false)
	return err
}

// runOutput is like run but returns the output of the command.
func (v *vcsCmd) runOutput(dir string, cmd string, keyval ...string) ([]byte, error) {
	return v.run1(dir, cmd, keyval, true)
}

// run1 is the generalized implementation of run and runOutput.
func (v *vcsCmd) run1(dir string, cmdline string, keyval []string, verbose bool) ([]byte, error) {
	m := make(map[string]string)
	for i := 0; i < len(keyval); i += 2 {
		m[keyval[i]] = keyval[i+1]
	}
	args := strings.Fields(cmdline)
	for i, arg := range args {
		args[i] = expand(m, arg)
	}

	_, err := exec.LookPath(v.cmd)
	if err != nil {
		fmt.Fprintf(os.Stderr,
			"go: missing %s command. See http://golang.org/s/gogetcmd\n",
			v.name)
		return nil, err
	}

	cmd := exec.Command(v.cmd, args...)
	cmd.Dir = dir
	cmd.Env = envForDir(cmd.Dir)
	if buildX {
		fmt.Printf("cd %s\n", dir)
		fmt.Printf("%s %s\n", v.cmd, strings.Join(args, " "))
	}
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err = cmd.Run()
	out := buf.Bytes()
	if err != nil {
		if verbose || buildV {
			fmt.Fprintf(os.Stderr, "# cd %s; %s %s\n", dir, v.cmd, strings.Join(args, " "))
			os.Stderr.Write(out)
		}
		return nil, err
	}
	return out, nil
}

// ping pings to determine scheme to use.
func (v *vcsCmd) ping(scheme, repo string) error {
	return v.runVerboseOnly(".", v.pingCmd, "scheme", scheme, "repo", repo)
}

// create creates a new copy of repo in dir.
// The parent of dir must exist; dir must not.
func (v *vcsCmd) create(dir, repo string) error {
	return v.run(".", v.createCmd, "dir", dir, "repo", repo)
}

// download downloads any new changes for the repo in dir.
func (v *vcsCmd) download(dir string) error {
	if err := v.fixDetachedHead(dir); err != nil {
		return err
	}
	return v.run(dir, v.downloadCmd)
}

// fixDetachedHead switches a Git repository in dir from a detached head to the master branch.
// Go versions before 1.2 downloaded Git repositories in an unfortunate way
// that resulted in the working tree state being on a detached head.
// That meant the repository was not usable for normal Git operations.
// Go 1.2 fixed that, but we can't pull into a detached head, so if this is
// a Git repository we check for being on a detached head and switch to the
// real branch, almost always called "master".
// TODO(dsymonds): Consider removing this for Go 1.3.
func (v *vcsCmd) fixDetachedHead(dir string) error {
	if v != vcsGit {
		return nil
	}

	// "git symbolic-ref HEAD" succeeds iff we are not on a detached head.
	if err := v.runVerboseOnly(dir, "symbolic-ref HEAD"); err == nil {
		// not on a detached head
		return nil
	}
	if buildV {
		log.Printf("%s on detached head; repairing", dir)
	}
	return v.run(dir, "checkout master")
}

// tags returns the list of available tags for the repo in dir.
func (v *vcsCmd) tags(dir string) ([]string, error) {
	var tags []string
	for _, tc := range v.tagCmd {
		out, err := v.runOutput(dir, tc.cmd)
		if err != nil {
			return nil, err
		}
		re := regexp.MustCompile(`(?m-s)` + tc.pattern)
		for _, m := range re.FindAllStringSubmatch(string(out), -1) {
			tags = append(tags, m[1])
		}
	}
	return tags, nil
}

// tagSync syncs the repo in dir to the named tag,
// which either is a tag returned by tags or is v.tagDefault.
func (v *vcsCmd) tagSync(dir, tag string) error {
	if v.tagSyncCmd == "" {
		return nil
	}
	if tag != "" {
		for _, tc := range v.tagLookupCmd {
			out, err := v.runOutput(dir, tc.cmd, "tag", tag)
			if err != nil {
				return err
			}
			re := regexp.MustCompile(`(?m-s)` + tc.pattern)
			m := re.FindStringSubmatch(string(out))
			if len(m) > 1 {
				tag = m[1]
				break
			}
		}
	}
	if tag == "" && v.tagSyncDefault != "" {
		return v.run(dir, v.tagSyncDefault)
	}
	return v.run(dir, v.tagSyncCmd, "tag", tag)
}

// A vcsPath describes how to convert an import path into a
// version control system and repository name.
type vcsPath struct {
	prefix string                              // prefix this description applies to
	re     string                              // pattern for import path
	repo   string                              // repository to use (expand with match of re)
	vcs    string                              // version control system to use (expand with match of re)
	check  func(match map[string]string) error // additional checks
	ping   bool                                // ping for scheme to use to download repo

	regexp *regexp.Regexp // cached compiled form of re
}

// vcsForDir inspects dir and its parents to determine the
// version control system and code repository to use.
// On return, root is the import path
// corresponding to the root of the repository
// (thus root is a prefix of importPath).
func vcsForDir(p *Package) (vcs *vcsCmd, root string, err error) {
	// Clean and double-check that dir is in (a subdirectory of) srcRoot.
	dir := filepath.Clean(p.Dir)
	srcRoot := filepath.Clean(p.build.SrcRoot)
	if len(dir) <= len(srcRoot) || dir[len(srcRoot)] != filepath.Separator {
		return nil, "", fmt.Errorf("directory %q is outside source root %q", dir, srcRoot)
	}

	origDir := dir
	for len(dir) > len(srcRoot) {
		for _, vcs := range vcsList {
			if fi, err := os.Stat(filepath.Join(dir, "."+vcs.cmd)); err == nil && fi.IsDir() {
				return vcs, dir[len(srcRoot)+1:], nil
			}
		}

		// Move to parent.
		ndir := filepath.Dir(dir)
		if len(ndir) >= len(dir) {
			// Shouldn't happen, but just in case, stop.
			break
		}
		dir = ndir
	}

	return nil, "", fmt.Errorf("directory %q is not using a known version control system", origDir)
}

// repoRoot represents a version control system, a repo, and a root of
// where to put it on disk.
type repoRoot struct {
	vcs *vcsCmd

	// repo is the repository URL, including scheme
	repo string

	// root is the import path corresponding to the root of the
	// repository
	root string
}

var httpPrefixRE = regexp.MustCompile(`^https?:`)

// repoRootForImportPath analyzes importPath to determine the
// version control system, and code repository to use.
func repoRootForImportPath(importPath string) (*repoRoot, error) {
	rr, err := repoRootForImportPathStatic(importPath, "")
	if err == errUnknownSite {
		// If there are wildcards, look up the thing before the wildcard,
		// hoping it applies to the wildcarded parts too.
		// This makes 'go get rsc.io/pdf/...' work in a fresh GOPATH.
		lookup := strings.TrimSuffix(importPath, "/...")
		if i := strings.Index(lookup, "/.../"); i >= 0 {
			lookup = lookup[:i]
		}
		rr, err = repoRootForImportDynamic(lookup)

		// repoRootForImportDynamic returns error detail
		// that is irrelevant if the user didn't intend to use a
		// dynamic import in the first place.
		// Squelch it.
		if err != nil {
			if buildV {
				log.Printf("import %q: %v", importPath, err)
			}
			err = fmt.Errorf("unrecognized import path %q", importPath)
		}
	}

	if err == nil && strings.Contains(importPath, "...") && strings.Contains(rr.root, "...") {
		// Do not allow wildcards in the repo root.
		rr = nil
		err = fmt.Errorf("cannot expand ... in %q", importPath)
	}
	return rr, err
}

var errUnknownSite = errors.New("dynamic lookup required to find mapping")

// repoRootForImportPathStatic attempts to map importPath to a
// repoRoot using the commonly-used VCS hosting sites in vcsPaths
// (github.com/user/dir), or from a fully-qualified importPath already
// containing its VCS type (foo.com/repo.git/dir)
//
// If scheme is non-empty, that scheme is forced.
func repoRootForImportPathStatic(importPath, scheme string) (*repoRoot, error) {
	// A common error is to use https://packagepath because that's what
	// hg and git require. Diagnose this helpfully.
	if loc := httpPrefixRE.FindStringIndex(importPath); loc != nil {
		// The importPath has been cleaned, so has only one slash. The pattern
		// ignores the slashes; the error message puts them back on the RHS at least.
		return nil, fmt.Errorf("%q not allowed in import path", importPath[loc[0]:loc[1]]+"//")
	}
	for _, srv := range vcsPaths {
		if !strings.HasPrefix(importPath, srv.prefix) {
			continue
		}
		m := srv.regexp.FindStringSubmatch(importPath)
		if m == nil {
			if srv.prefix != "" {
				return nil, fmt.Errorf("invalid %s import path %q", srv.prefix, importPath)
			}
			continue
		}

		// Build map of named subexpression matches for expand.
		match := map[string]string{
			"prefix": srv.prefix,
			"import": importPath,
		}
		for i, name := range srv.regexp.SubexpNames() {
			if name != "" && match[name] == "" {
				match[name] = m[i]
			}
		}
		if srv.vcs != "" {
			match["vcs"] = expand(match, srv.vcs)
		}
		if srv.repo != "" {
			match["repo"] = expand(match, srv.repo)
		}
		if srv.check != nil {
			if err := srv.check(match); err != nil {
				return nil, err
			}
		}
		vcs := vcsByCmd(match["vcs"])
		if vcs == nil {
			return nil, fmt.Errorf("unknown version control system %q", match["vcs"])
		}
		if srv.ping {
			if scheme != "" {
				match["repo"] = scheme + "://" + match["repo"]
			} else {
				for _, scheme := range vcs.scheme {
					if vcs.ping(scheme, match["repo"]) == nil {
						match["repo"] = scheme + "://" + match["repo"]
						break
					}
				}
			}
		}
		rr := &repoRoot{
			vcs:  vcs,
			repo: match["repo"],
			root: match["root"],
		}
		return rr, nil
	}
	return nil, errUnknownSite
}

// repoRootForImportDynamic finds a *repoRoot for a custom domain that's not
// statically known by repoRootForImportPathStatic.
//
// This handles "vanity import paths" like "name.tld/pkg/foo".
func repoRootForImportDynamic(importPath string) (*repoRoot, error) {
	slash := strings.Index(importPath, "/")
	if slash < 0 {
		return nil, errors.New("import path does not contain a slash")
	}
	host := importPath[:slash]
	if !strings.Contains(host, ".") {
		return nil, errors.New("import path does not begin with hostname")
	}
	urlStr, body, err := httpsOrHTTP(importPath)
	if err != nil {
		return nil, fmt.Errorf("http/https fetch: %v", err)
	}
	defer body.Close()
	imports, err := parseMetaGoImports(body)
	if err != nil {
		return nil, fmt.Errorf("parsing %s: %v", importPath, err)
	}
	metaImport, err := matchGoImport(imports, importPath)
	if err != nil {
		if err != errNoMatch {
			return nil, fmt.Errorf("parse %s: %v", urlStr, err)
		}
		return nil, fmt.Errorf("parse %s: no go-import meta tags", urlStr)
	}
	if buildV {
		log.Printf("get %q: found meta tag %#v at %s", importPath, metaImport, urlStr)
	}
	// If the import was "uni.edu/bob/project", which said the
	// prefix was "uni.edu" and the RepoRoot was "evilroot.com",
	// make sure we don't trust Bob and check out evilroot.com to
	// "uni.edu" yet (possibly overwriting/preempting another
	// non-evil student).  Instead, first verify the root and see
	// if it matches Bob's claim.
	if metaImport.Prefix != importPath {
		if buildV {
			log.Printf("get %q: verifying non-authoritative meta tag", importPath)
		}
		urlStr0 := urlStr
		urlStr, body, err = httpsOrHTTP(metaImport.Prefix)
		if err != nil {
			return nil, fmt.Errorf("fetch %s: %v", urlStr, err)
		}
		imports, err := parseMetaGoImports(body)
		if err != nil {
			return nil, fmt.Errorf("parsing %s: %v", importPath, err)
		}
		if len(imports) == 0 {
			return nil, fmt.Errorf("fetch %s: no go-import meta tag", urlStr)
		}
		metaImport2, err := matchGoImport(imports, importPath)
		if err != nil || metaImport != metaImport2 {
			return nil, fmt.Errorf("%s and %s disagree about go-import for %s", urlStr0, urlStr, metaImport.Prefix)
		}
	}

	if !strings.Contains(metaImport.RepoRoot, "://") {
		return nil, fmt.Errorf("%s: invalid repo root %q; no scheme", urlStr, metaImport.RepoRoot)
	}
	rr := &repoRoot{
		vcs:  vcsByCmd(metaImport.VCS),
		repo: metaImport.RepoRoot,
		root: metaImport.Prefix,
	}
	if rr.vcs == nil {
		return nil, fmt.Errorf("%s: unknown vcs %q", urlStr, metaImport.VCS)
	}
	return rr, nil
}

// metaImport represents the parsed <meta name="go-import"
// content="prefix vcs reporoot" /> tags from HTML files.
type metaImport struct {
	Prefix, VCS, RepoRoot string
}

// errNoMatch is returned from matchGoImport when there's no applicable match.
var errNoMatch = errors.New("no import match")

// matchGoImport returns the metaImport from imports matching importPath.
// An error is returned if there are multiple matches.
// errNoMatch is returned if none match.
func matchGoImport(imports []metaImport, importPath string) (_ metaImport, err error) {
	match := -1
	for i, im := range imports {
		if !strings.HasPrefix(importPath, im.Prefix) {
			continue
		}
		if match != -1 {
			err = fmt.Errorf("multiple meta tags match import path %q", importPath)
			return
		}
		match = i
	}
	if match == -1 {
		err = errNoMatch
		return
	}
	return imports[match], nil
}

// expand rewrites s to replace {k} with match[k] for each key k in match.
func expand(match map[string]string, s string) string {
	for k, v := range match {
		s = strings.Replace(s, "{"+k+"}", v, -1)
	}
	return s
}

// vcsPaths lists the known vcs paths.
var vcsPaths = []*vcsPath{
	// Google Code - new syntax
	{
		prefix: "code.google.com/",
		re:     `^(?P<root>code\.google\.com/p/(?P<project>[a-z0-9\-]+)(\.(?P<subrepo>[a-z0-9\-]+))?)(/[A-Za-z0-9_.\-]+)*$`,
		repo:   "https://{root}",
		check:  googleCodeVCS,
	},

	// Google Code - old syntax
	{
		re:    `^(?P<project>[a-z0-9_\-.]+)\.googlecode\.com/(git|hg|svn)(?P<path>/.*)?$`,
		check: oldGoogleCode,
	},

	// Github
	{
		prefix: "github.com/",
		re:     `^(?P<root>github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)(/[A-Za-z0-9_.\-]+)*$`,
		vcs:    "git",
		repo:   "https://{root}",
		check:  noVCSSuffix,
	},

	// Bitbucket
	{
		prefix: "bitbucket.org/",
		re:     `^(?P<root>bitbucket\.org/(?P<bitname>[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`,
		repo:   "https://{root}",
		check:  bitbucketVCS,
	},

	// Launchpad
	{
		prefix: "launchpad.net/",
		re:     `^(?P<root>launchpad\.net/((?P<project>[A-Za-z0-9_.\-]+)(?P<series>/[A-Za-z0-9_.\-]+)?|~[A-Za-z0-9_.\-]+/(\+junk|[A-Za-z0-9_.\-]+)/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`,
		vcs:    "bzr",
		repo:   "https://{root}",
		check:  launchpadVCS,
	},

	// IBM DevOps Services (JazzHub)
	{
		prefix: "hub.jazz.net/git",
		re:     `^(?P<root>hub.jazz.net/git/[a-z0-9]+/[A-Za-z0-9_.\-]+)(/[A-Za-z0-9_.\-]+)*$`,
		vcs:    "git",
		repo:   "https://{root}",
		check:  noVCSSuffix,
	},

	// General syntax for any server.
	{
		re:   `^(?P<root>(?P<repo>([a-z0-9.\-]+\.)+[a-z0-9.\-]+(:[0-9]+)?/[A-Za-z0-9_.\-/]*?)\.(?P<vcs>bzr|git|hg|svn))(/[A-Za-z0-9_.\-]+)*$`,
		ping: true,
	},
}

func init() {
	// fill in cached regexps.
	// Doing this eagerly discovers invalid regexp syntax
	// without having to run a command that needs that regexp.
	for _, srv := range vcsPaths {
		srv.regexp = regexp.MustCompile(srv.re)
	}
}

// noVCSSuffix checks that the repository name does not
// end in .foo for any version control system foo.
// The usual culprit is ".git".
func noVCSSuffix(match map[string]string) error {
	repo := match["repo"]
	for _, vcs := range vcsList {
		if strings.HasSuffix(repo, "."+vcs.cmd) {
			return fmt.Errorf("invalid version control suffix in %s path", match["prefix"])
		}
	}
	return nil
}

var googleCheckout = regexp.MustCompile(`id="checkoutcmd">(hg|git|svn)`)

// googleCodeVCS determines the version control system for
// a code.google.com repository, by scraping the project's
// /source/checkout page.
func googleCodeVCS(match map[string]string) error {
	if err := noVCSSuffix(match); err != nil {
		return err
	}
	data, err := httpGET(expand(match, "https://code.google.com/p/{project}/source/checkout?repo={subrepo}"))
	if err != nil {
		return err
	}

	if m := googleCheckout.FindSubmatch(data); m != nil {
		if vcs := vcsByCmd(string(m[1])); vcs != nil {
			// Subversion requires the old URLs.
			// TODO: Test.
			if vcs == vcsSvn {
				if match["subrepo"] != "" {
					return fmt.Errorf("sub-repositories not supported in Google Code Subversion projects")
				}
				match["repo"] = expand(match, "https://{project}.googlecode.com/svn")
			}
			match["vcs"] = vcs.cmd
			return nil
		}
	}

	return fmt.Errorf("unable to detect version control system for code.google.com/ path")
}

// oldGoogleCode is invoked for old-style foo.googlecode.com paths.
// It prints an error giving the equivalent new path.
func oldGoogleCode(match map[string]string) error {
	return fmt.Errorf("invalid Google Code import path: use %s instead",
		expand(match, "code.google.com/p/{project}{path}"))
}

// bitbucketVCS determines the version control system for a
// Bitbucket repository, by using the Bitbucket API.
func bitbucketVCS(match map[string]string) error {
	if err := noVCSSuffix(match); err != nil {
		return err
	}

	var resp struct {
		SCM string `json:"scm"`
	}
	url := expand(match, "https://api.bitbucket.org/1.0/repositories/{bitname}")
	data, err := httpGET(url)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return fmt.Errorf("decoding %s: %v", url, err)
	}

	if vcsByCmd(resp.SCM) != nil {
		match["vcs"] = resp.SCM
		if resp.SCM == "git" {
			match["repo"] += ".git"
		}
		return nil
	}

	return fmt.Errorf("unable to detect version control system for bitbucket.org/ path")
}

// launchpadVCS solves the ambiguity for "lp.net/project/foo". In this case,
// "foo" could be a series name registered in Launchpad with its own branch,
// and it could also be the name of a directory within the main project
// branch one level up.
func launchpadVCS(match map[string]string) error {
	if match["project"] == "" || match["series"] == "" {
		return nil
	}
	_, err := httpGET(expand(match, "https://code.launchpad.net/{project}{series}/.bzr/branch-format"))
	if err != nil {
		match["root"] = expand(match, "launchpad.net/{project}")
		match["repo"] = expand(match, "https://{root}")
	}
	return nil
}
