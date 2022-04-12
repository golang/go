// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vcs exposes functions for resolving import paths
// and using version control systems, which can be used to
// implement behavior similar to the standard "go get" command.
//
// This package is a copy of internal code in package cmd/go/internal/get,
// modified to make the identifiers exported. It's provided here
// for developers who want to write tools with similar semantics.
// It needs to be manually kept in sync with upstream when changes are
// made to cmd/go/internal/get; see https://golang.org/issue/11490.
package vcs // import "golang.org/x/tools/go/vcs"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	exec "golang.org/x/sys/execabs"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Verbose enables verbose operation logging.
var Verbose bool

// ShowCmd controls whether VCS commands are printed.
var ShowCmd bool

// A Cmd describes how to use a version control system
// like Mercurial, Git, or Subversion.
type Cmd struct {
	Name string
	Cmd  string // name of binary to invoke command

	CreateCmd   string // command to download a fresh copy of a repository
	DownloadCmd string // command to download updates into an existing repository

	TagCmd         []TagCmd // commands to list tags
	TagLookupCmd   []TagCmd // commands to lookup tags before running tagSyncCmd
	TagSyncCmd     string   // command to sync to specific tag
	TagSyncDefault string   // command to sync to default tag

	LogCmd string // command to list repository changelogs in an XML format

	Scheme  []string
	PingCmd string
}

// A TagCmd describes a command to list available tags
// that can be passed to Cmd.TagSyncCmd.
type TagCmd struct {
	Cmd     string // command to list tags
	Pattern string // regexp to extract tags from list
}

// vcsList lists the known version control systems
var vcsList = []*Cmd{
	vcsHg,
	vcsGit,
	vcsSvn,
	vcsBzr,
}

// ByCmd returns the version control system for the given
// command name (hg, git, svn, bzr).
func ByCmd(cmd string) *Cmd {
	for _, vcs := range vcsList {
		if vcs.Cmd == cmd {
			return vcs
		}
	}
	return nil
}

// vcsHg describes how to use Mercurial.
var vcsHg = &Cmd{
	Name: "Mercurial",
	Cmd:  "hg",

	CreateCmd:   "clone -U {repo} {dir}",
	DownloadCmd: "pull",

	// We allow both tag and branch names as 'tags'
	// for selecting a version.  This lets people have
	// a go.release.r60 branch and a go1 branch
	// and make changes in both, without constantly
	// editing .hgtags.
	TagCmd: []TagCmd{
		{"tags", `^(\S+)`},
		{"branches", `^(\S+)`},
	},
	TagSyncCmd:     "update -r {tag}",
	TagSyncDefault: "update default",

	LogCmd: "log --encoding=utf-8 --limit={limit} --template={template}",

	Scheme:  []string{"https", "http", "ssh"},
	PingCmd: "identify {scheme}://{repo}",
}

// vcsGit describes how to use Git.
var vcsGit = &Cmd{
	Name: "Git",
	Cmd:  "git",

	CreateCmd:   "clone {repo} {dir}",
	DownloadCmd: "pull --ff-only",

	TagCmd: []TagCmd{
		// tags/xxx matches a git tag named xxx
		// origin/xxx matches a git branch named xxx on the default remote repository
		{"show-ref", `(?:tags|origin)/(\S+)$`},
	},
	TagLookupCmd: []TagCmd{
		{"show-ref tags/{tag} origin/{tag}", `((?:tags|origin)/\S+)$`},
	},
	TagSyncCmd:     "checkout {tag}",
	TagSyncDefault: "checkout master",

	Scheme:  []string{"git", "https", "http", "git+ssh"},
	PingCmd: "ls-remote {scheme}://{repo}",
}

// vcsBzr describes how to use Bazaar.
var vcsBzr = &Cmd{
	Name: "Bazaar",
	Cmd:  "bzr",

	CreateCmd: "branch {repo} {dir}",

	// Without --overwrite bzr will not pull tags that changed.
	// Replace by --overwrite-tags after http://pad.lv/681792 goes in.
	DownloadCmd: "pull --overwrite",

	TagCmd:         []TagCmd{{"tags", `^(\S+)`}},
	TagSyncCmd:     "update -r {tag}",
	TagSyncDefault: "update -r revno:-1",

	Scheme:  []string{"https", "http", "bzr", "bzr+ssh"},
	PingCmd: "info {scheme}://{repo}",
}

// vcsSvn describes how to use Subversion.
var vcsSvn = &Cmd{
	Name: "Subversion",
	Cmd:  "svn",

	CreateCmd:   "checkout {repo} {dir}",
	DownloadCmd: "update",

	// There is no tag command in subversion.
	// The branch information is all in the path names.

	LogCmd: "log --xml --limit={limit}",

	Scheme:  []string{"https", "http", "svn", "svn+ssh"},
	PingCmd: "info {scheme}://{repo}",
}

func (v *Cmd) String() string {
	return v.Name
}

// run runs the command line cmd in the given directory.
// keyval is a list of key, value pairs.  run expands
// instances of {key} in cmd into value, but only after
// splitting cmd into individual arguments.
// If an error occurs, run prints the command line and the
// command's combined stdout+stderr to standard error.
// Otherwise run discards the command's output.
func (v *Cmd) run(dir string, cmd string, keyval ...string) error {
	_, err := v.run1(dir, cmd, keyval, true)
	return err
}

// runVerboseOnly is like run but only generates error output to standard error in verbose mode.
func (v *Cmd) runVerboseOnly(dir string, cmd string, keyval ...string) error {
	_, err := v.run1(dir, cmd, keyval, false)
	return err
}

// runOutput is like run but returns the output of the command.
func (v *Cmd) runOutput(dir string, cmd string, keyval ...string) ([]byte, error) {
	return v.run1(dir, cmd, keyval, true)
}

// run1 is the generalized implementation of run and runOutput.
func (v *Cmd) run1(dir string, cmdline string, keyval []string, verbose bool) ([]byte, error) {
	m := make(map[string]string)
	for i := 0; i < len(keyval); i += 2 {
		m[keyval[i]] = keyval[i+1]
	}
	args := strings.Fields(cmdline)
	for i, arg := range args {
		args[i] = expand(m, arg)
	}

	_, err := exec.LookPath(v.Cmd)
	if err != nil {
		fmt.Fprintf(os.Stderr,
			"go: missing %s command. See http://golang.org/s/gogetcmd\n",
			v.Name)
		return nil, err
	}

	cmd := exec.Command(v.Cmd, args...)
	cmd.Dir = dir
	cmd.Env = envForDir(cmd.Dir)
	if ShowCmd {
		fmt.Printf("cd %s\n", dir)
		fmt.Printf("%s %s\n", v.Cmd, strings.Join(args, " "))
	}
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err = cmd.Run()
	out := buf.Bytes()
	if err != nil {
		if verbose || Verbose {
			fmt.Fprintf(os.Stderr, "# cd %s; %s %s\n", dir, v.Cmd, strings.Join(args, " "))
			os.Stderr.Write(out)
		}
		return nil, err
	}
	return out, nil
}

// Ping pings the repo to determine if scheme used is valid.
// This repo must be pingable with this scheme and VCS.
func (v *Cmd) Ping(scheme, repo string) error {
	return v.runVerboseOnly(".", v.PingCmd, "scheme", scheme, "repo", repo)
}

// Create creates a new copy of repo in dir.
// The parent of dir must exist; dir must not.
func (v *Cmd) Create(dir, repo string) error {
	return v.run(".", v.CreateCmd, "dir", dir, "repo", repo)
}

// CreateAtRev creates a new copy of repo in dir at revision rev.
// The parent of dir must exist; dir must not.
// rev must be a valid revision in repo.
func (v *Cmd) CreateAtRev(dir, repo, rev string) error {
	if err := v.Create(dir, repo); err != nil {
		return err
	}
	return v.run(dir, v.TagSyncCmd, "tag", rev)
}

// Download downloads any new changes for the repo in dir.
// dir must be a valid VCS repo compatible with v.
func (v *Cmd) Download(dir string) error {
	return v.run(dir, v.DownloadCmd)
}

// Tags returns the list of available tags for the repo in dir.
// dir must be a valid VCS repo compatible with v.
func (v *Cmd) Tags(dir string) ([]string, error) {
	var tags []string
	for _, tc := range v.TagCmd {
		out, err := v.runOutput(dir, tc.Cmd)
		if err != nil {
			return nil, err
		}
		re := regexp.MustCompile(`(?m-s)` + tc.Pattern)
		for _, m := range re.FindAllStringSubmatch(string(out), -1) {
			tags = append(tags, m[1])
		}
	}
	return tags, nil
}

// TagSync syncs the repo in dir to the named tag, which is either a
// tag returned by Tags or the empty string (the default tag).
// dir must be a valid VCS repo compatible with v and the tag must exist.
func (v *Cmd) TagSync(dir, tag string) error {
	if v.TagSyncCmd == "" {
		return nil
	}
	if tag != "" {
		for _, tc := range v.TagLookupCmd {
			out, err := v.runOutput(dir, tc.Cmd, "tag", tag)
			if err != nil {
				return err
			}
			re := regexp.MustCompile(`(?m-s)` + tc.Pattern)
			m := re.FindStringSubmatch(string(out))
			if len(m) > 1 {
				tag = m[1]
				break
			}
		}
	}
	if tag == "" && v.TagSyncDefault != "" {
		return v.run(dir, v.TagSyncDefault)
	}
	return v.run(dir, v.TagSyncCmd, "tag", tag)
}

// Log logs the changes for the repo in dir.
// dir must be a valid VCS repo compatible with v.
func (v *Cmd) Log(dir, logTemplate string) ([]byte, error) {
	if err := v.Download(dir); err != nil {
		return []byte{}, err
	}

	const N = 50 // how many revisions to grab
	return v.runOutput(dir, v.LogCmd, "limit", strconv.Itoa(N), "template", logTemplate)
}

// LogAtRev logs the change for repo in dir at the rev revision.
// dir must be a valid VCS repo compatible with v.
// rev must be a valid revision for the repo in dir.
func (v *Cmd) LogAtRev(dir, rev, logTemplate string) ([]byte, error) {
	if err := v.Download(dir); err != nil {
		return []byte{}, err
	}

	// Append revision flag to LogCmd.
	logAtRevCmd := v.LogCmd + " --rev=" + rev
	return v.runOutput(dir, logAtRevCmd, "limit", strconv.Itoa(1), "template", logTemplate)
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

// FromDir inspects dir and its parents to determine the
// version control system and code repository to use.
// On return, root is the import path
// corresponding to the root of the repository.
func FromDir(dir, srcRoot string) (vcs *Cmd, root string, err error) {
	// Clean and double-check that dir is in (a subdirectory of) srcRoot.
	dir = filepath.Clean(dir)
	srcRoot = filepath.Clean(srcRoot)
	if len(dir) <= len(srcRoot) || dir[len(srcRoot)] != filepath.Separator {
		return nil, "", fmt.Errorf("directory %q is outside source root %q", dir, srcRoot)
	}

	var vcsRet *Cmd
	var rootRet string

	origDir := dir
	for len(dir) > len(srcRoot) {
		for _, vcs := range vcsList {
			if _, err := os.Stat(filepath.Join(dir, "."+vcs.Cmd)); err == nil {
				root := filepath.ToSlash(dir[len(srcRoot)+1:])
				// Record first VCS we find, but keep looking,
				// to detect mistakes like one kind of VCS inside another.
				if vcsRet == nil {
					vcsRet = vcs
					rootRet = root
					continue
				}
				// Allow .git inside .git, which can arise due to submodules.
				if vcsRet == vcs && vcs.Cmd == "git" {
					continue
				}
				// Otherwise, we have one VCS inside a different VCS.
				return nil, "", fmt.Errorf("directory %q uses %s, but parent %q uses %s",
					filepath.Join(srcRoot, rootRet), vcsRet.Cmd, filepath.Join(srcRoot, root), vcs.Cmd)
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

	if vcsRet != nil {
		return vcsRet, rootRet, nil
	}

	return nil, "", fmt.Errorf("directory %q is not using a known version control system", origDir)
}

// RepoRoot represents a version control system, a repo, and a root of
// where to put it on disk.
type RepoRoot struct {
	VCS *Cmd

	// Repo is the repository URL, including scheme.
	Repo string

	// Root is the import path corresponding to the root of the
	// repository.
	Root string
}

// RepoRootForImportPath analyzes importPath to determine the
// version control system, and code repository to use.
func RepoRootForImportPath(importPath string, verbose bool) (*RepoRoot, error) {
	rr, err := RepoRootForImportPathStatic(importPath, "")
	if err == errUnknownSite {
		rr, err = RepoRootForImportDynamic(importPath, verbose)

		// RepoRootForImportDynamic returns error detail
		// that is irrelevant if the user didn't intend to use a
		// dynamic import in the first place.
		// Squelch it.
		if err != nil {
			if Verbose {
				log.Printf("import %q: %v", importPath, err)
			}
			err = fmt.Errorf("unrecognized import path %q", importPath)
		}
	}

	if err == nil && strings.Contains(importPath, "...") && strings.Contains(rr.Root, "...") {
		// Do not allow wildcards in the repo root.
		rr = nil
		err = fmt.Errorf("cannot expand ... in %q", importPath)
	}
	return rr, err
}

var errUnknownSite = errors.New("dynamic lookup required to find mapping")

// RepoRootForImportPathStatic attempts to map importPath to a
// RepoRoot using the commonly-used VCS hosting sites in vcsPaths
// (github.com/user/dir), or from a fully-qualified importPath already
// containing its VCS type (foo.com/repo.git/dir)
//
// If scheme is non-empty, that scheme is forced.
func RepoRootForImportPathStatic(importPath, scheme string) (*RepoRoot, error) {
	if strings.Contains(importPath, "://") {
		return nil, fmt.Errorf("invalid import path %q", importPath)
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
		vcs := ByCmd(match["vcs"])
		if vcs == nil {
			return nil, fmt.Errorf("unknown version control system %q", match["vcs"])
		}
		if srv.ping {
			if scheme != "" {
				match["repo"] = scheme + "://" + match["repo"]
			} else {
				for _, scheme := range vcs.Scheme {
					if vcs.Ping(scheme, match["repo"]) == nil {
						match["repo"] = scheme + "://" + match["repo"]
						break
					}
				}
			}
		}
		rr := &RepoRoot{
			VCS:  vcs,
			Repo: match["repo"],
			Root: match["root"],
		}
		return rr, nil
	}
	return nil, errUnknownSite
}

// RepoRootForImportDynamic finds a *RepoRoot for a custom domain that's not
// statically known by RepoRootForImportPathStatic.
//
// This handles custom import paths like "name.tld/pkg/foo" or just "name.tld".
func RepoRootForImportDynamic(importPath string, verbose bool) (*RepoRoot, error) {
	slash := strings.Index(importPath, "/")
	if slash < 0 {
		slash = len(importPath)
	}
	host := importPath[:slash]
	if !strings.Contains(host, ".") {
		return nil, errors.New("import path doesn't contain a hostname")
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
	if verbose {
		log.Printf("get %q: found meta tag %#v at %s", importPath, metaImport, urlStr)
	}
	// If the import was "uni.edu/bob/project", which said the
	// prefix was "uni.edu" and the RepoRoot was "evilroot.com",
	// make sure we don't trust Bob and check out evilroot.com to
	// "uni.edu" yet (possibly overwriting/preempting another
	// non-evil student).  Instead, first verify the root and see
	// if it matches Bob's claim.
	if metaImport.Prefix != importPath {
		if verbose {
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

	if err := validateRepoRoot(metaImport.RepoRoot); err != nil {
		return nil, fmt.Errorf("%s: invalid repo root %q: %v", urlStr, metaImport.RepoRoot, err)
	}
	rr := &RepoRoot{
		VCS:  ByCmd(metaImport.VCS),
		Repo: metaImport.RepoRoot,
		Root: metaImport.Prefix,
	}
	if rr.VCS == nil {
		return nil, fmt.Errorf("%s: unknown vcs %q", urlStr, metaImport.VCS)
	}
	return rr, nil
}

// validateRepoRoot returns an error if repoRoot does not seem to be
// a valid URL with scheme.
func validateRepoRoot(repoRoot string) error {
	url, err := url.Parse(repoRoot)
	if err != nil {
		return err
	}
	if url.Scheme == "" {
		return errors.New("no scheme")
	}
	return nil
}

// metaImport represents the parsed <meta name="go-import"
// content="prefix vcs reporoot" /> tags from HTML files.
type metaImport struct {
	Prefix, VCS, RepoRoot string
}

// errNoMatch is returned from matchGoImport when there's no applicable match.
var errNoMatch = errors.New("no import match")

// pathPrefix reports whether sub is a prefix of s,
// only considering entire path components.
func pathPrefix(s, sub string) bool {
	// strings.HasPrefix is necessary but not sufficient.
	if !strings.HasPrefix(s, sub) {
		return false
	}
	// The remainder after the prefix must either be empty or start with a slash.
	rem := s[len(sub):]
	return rem == "" || rem[0] == '/'
}

// matchGoImport returns the metaImport from imports matching importPath.
// An error is returned if there are multiple matches.
// errNoMatch is returned if none match.
func matchGoImport(imports []metaImport, importPath string) (_ metaImport, err error) {
	match := -1
	for i, im := range imports {
		if !pathPrefix(importPath, im.Prefix) {
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
	// Github
	{
		prefix: "github.com/",
		re:     `^(?P<root>github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)(/[\p{L}0-9_.\-]+)*$`,
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

	// Git at OpenStack
	{
		prefix: "git.openstack.org",
		re:     `^(?P<root>git\.openstack\.org/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)(\.git)?(/[A-Za-z0-9_.\-]+)*$`,
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
		if strings.HasSuffix(repo, "."+vcs.Cmd) {
			return fmt.Errorf("invalid version control suffix in %s path", match["prefix"])
		}
	}
	return nil
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
	url := expand(match, "https://api.bitbucket.org/2.0/repositories/{bitname}?fields=scm")
	data, err := httpGET(url)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return fmt.Errorf("decoding %s: %v", url, err)
	}

	if ByCmd(resp.SCM) != nil {
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
