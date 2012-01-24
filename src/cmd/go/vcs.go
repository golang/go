// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
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

	tagCmd     []tagCmd // commands to list tags
	tagDefault string   // default tag to use
	tagSyncCmd string   // command to sync to specific tag
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
	// a go.release.r60 branch and a go.1 branch
	// and make changes in both, without constantly
	// editing .hgtags.
	tagCmd: []tagCmd{
		{"tags", `^(\S+)`},
		{"branches", `^(\S+)`},
	},
	tagDefault: "default",
	tagSyncCmd: "update -r {tag}",
}

// vcsGit describes how to use Git.
var vcsGit = &vcsCmd{
	name: "Git",
	cmd:  "git",

	createCmd:   "clone {repo} {dir}",
	downloadCmd: "fetch",

	tagCmd:     []tagCmd{{"tag", `^(\S+)$`}},
	tagDefault: "master",
	tagSyncCmd: "checkout {tag}",
}

// vcsBzr describes how to use Bazaar.
var vcsBzr = &vcsCmd{
	name: "Bazaar",
	cmd:  "bzr",

	createCmd:   "branch {repo} {dir}",
	downloadCmd: "pull --overwrite", // TODO: REALLY?

	tagCmd:     []tagCmd{{"tags", `^(\S+)`}},
	tagDefault: "revno:-1",
	tagSyncCmd: "update -r {tag}",
}

// vcsSvn describes how to use Subversion.
var vcsSvn = &vcsCmd{
	name: "Subversion",
	cmd:  "svn",

	createCmd:   "checkout {repo} {dir}",
	downloadCmd: "update",

	// There is no tag command in subversion.
	// The branch information is all in the path names.
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
	_, err := v.run1(dir, false, cmd, keyval)
	return err
}

// runOutput is like run but returns the output of the command.
func (v *vcsCmd) runOutput(dir string, cmd string, keyval ...string) ([]byte, error) {
	return v.run1(dir, true, cmd, keyval)
}

// run1 is the generalized implementation of run and runOutput.
func (v *vcsCmd) run1(dir string, output bool, cmdline string, keyval []string) ([]byte, error) {
	m := make(map[string]string)
	for i := 0; i < len(keyval); i += 2 {
		m[keyval[i]] = keyval[i+1]
	}
	args := strings.Fields(cmdline)
	for i, arg := range args {
		args[i] = expand(m, arg)
	}

	cmd := exec.Command(v.cmd, args...)
	cmd.Dir = dir
	if buildX {
		fmt.Printf("cd %s\n", dir)
		fmt.Printf("%s %s\n", v.cmd, strings.Join(args, " "))
	}
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	out := buf.Bytes()
	err := cmd.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "# cd %s; %s %s\n", dir, v.cmd, strings.Join(args, " "))
		os.Stderr.Write(out)
		return nil, err
	}
	return out, nil
}

// create creates a new copy of repo in dir.
// The parent of dir must exist; dir must not.
func (v *vcsCmd) create(dir, repo string) error {
	return v.run(".", v.createCmd, "dir", dir, "repo", repo)
}

// download downloads any new changes for the repo in dir.
func (v *vcsCmd) download(dir string) error {
	return v.run(dir, v.downloadCmd)
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
		tags = append(tags, re.FindAllString(string(out), -1)...)
	}
	return tags, nil
}

// tagSync syncs the repo in dir to the named tag,
// which either is a tag returned by tags or is v.tagDefault.
func (v *vcsCmd) tagSync(dir, tag string) error {
	if v.tagSyncCmd == "" {
		return nil
	}
	return v.run(dir, v.tagSyncCmd, "tag", tag)
}

// A vcsPath is describes how to convert an import path into a
// version control system and repository name.
type vcsPath struct {
	prefix string                              // prefix this description applies to
	re     string                              // pattern for import path
	repo   string                              // repository to use (expand with match of re)
	vcs    string                              // version control system to use (expand with match of re)
	check  func(match map[string]string) error // additional checks

	regexp *regexp.Regexp // cached compiled form of re
}

// vcsForImportPath analyzes importPath to determine the
// version control system, and code repository to use.
// On return, repo is the repository URL and root is the
// import path corresponding to the root of the repository
// (thus root is a prefix of importPath).
func vcsForImportPath(importPath string) (vcs *vcsCmd, repo, root string, err error) {
	for _, srv := range vcsPaths {
		if !strings.HasPrefix(importPath, srv.prefix) {
			continue
		}
		m := srv.regexp.FindStringSubmatch(importPath)
		if m == nil {
			if srv.prefix != "" {
				return nil, "", "", fmt.Errorf("invalid %s import path %q", srv.prefix, importPath)
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
				return nil, "", "", err
			}
		}
		vcs := vcsByCmd(match["vcs"])
		if vcs == nil {
			return nil, "", "", fmt.Errorf("unknown version control system %q", match["vcs"])
		}
		return vcs, match["repo"], match["root"], nil
	}
	return nil, "", "", fmt.Errorf("unrecognized import path %q", importPath)
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
		re:     `^(?P<root>launchpad\.net/([A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)?|~[A-Za-z0-9_.\-]+/(\+junk|[A-Za-z0-9_.\-]+)/[A-Za-z0-9_.\-]+))(/[A-Za-z0-9_.\-]+)*$`,
		vcs:    "bzr",
		repo:   "https://{root}",
	},

	// General syntax for any server.
	{
		re: `^(?P<root>(?P<repo>([a-z0-9.\-]+\.)+[a-z0-9.\-]+(:[0-9]+)?/[A-Za-z0-9_.\-/]*?)\.(?P<vcs>bzr|git|hg|svn))(/[A-Za-z0-9_.\-]+)*$`,
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
// BitBucket repository, by using the BitBucket API.
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
