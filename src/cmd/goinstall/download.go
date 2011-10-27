// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Download remote packages.

package main

import (
	"bytes"
	"exec"
	"fmt"
	"http"
	"json"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

const dashboardURL = "http://godashboard.appspot.com/package"

// maybeReportToDashboard reports path to dashboard unless
// -dashboard=false is on command line.  It ignores errors.
func maybeReportToDashboard(path string) {
	// if -dashboard=false was on command line, do nothing
	if !*reportToDashboard {
		return
	}

	// otherwise lob url to dashboard
	r, _ := http.Post(dashboardURL, "application/x-www-form-urlencoded", strings.NewReader("path="+path))
	if r != nil && r.Body != nil {
		r.Body.Close()
	}
}

// a vcs represents a version control system
// like Mercurial, Git, or Subversion.
type vcs struct {
	name          string
	cmd           string
	metadir       string
	checkout      string
	clone         string
	update        string
	updateRevFlag string
	pull          string
	pullForceFlag string
	tagList       string
	tagListRe     *regexp.Regexp
	check         string
	protocols     []string
	suffix        string
	defaultHosts  []host
}

var hg = vcs{
	name:      "Mercurial",
	cmd:       "hg",
	metadir:   ".hg",
	checkout:  "checkout",
	clone:     "clone",
	update:    "update",
	pull:      "pull",
	tagList:   "tags",
	tagListRe: regexp.MustCompile("([^ ]+)[^\n]+\n"),
	check:     "identify",
	protocols: []string{"https", "http"},
	suffix:    ".hg",
}

var git = vcs{
	name:      "Git",
	cmd:       "git",
	metadir:   ".git",
	checkout:  "checkout",
	clone:     "clone",
	update:    "pull",
	pull:      "fetch",
	tagList:   "tag",
	tagListRe: regexp.MustCompile("([^\n]+)\n"),
	check:     "ls-remote",
	protocols: []string{"git", "https", "http"},
	suffix:    ".git",
}

var svn = vcs{
	name:      "Subversion",
	cmd:       "svn",
	metadir:   ".svn",
	checkout:  "checkout",
	clone:     "checkout",
	update:    "update",
	check:     "info",
	protocols: []string{"https", "http", "svn"},
	suffix:    ".svn",
}

var bzr = vcs{
	name:          "Bazaar",
	cmd:           "bzr",
	metadir:       ".bzr",
	checkout:      "update",
	clone:         "branch",
	update:        "update",
	updateRevFlag: "-r",
	pull:          "pull",
	pullForceFlag: "--overwrite",
	tagList:       "tags",
	tagListRe:     regexp.MustCompile("([^ ]+)[^\n]+\n"),
	check:         "info",
	protocols:     []string{"https", "http", "bzr"},
	suffix:        ".bzr",
}

var vcsList = []*vcs{&git, &hg, &bzr, &svn}

type host struct {
	pattern *regexp.Regexp
	getVcs  func(repo, path string) (*vcsMatch, os.Error)
}

var knownHosts = []host{
	{
		regexp.MustCompile(`^([a-z0-9\-]+\.googlecode\.com/(svn|git|hg))(/[a-z0-9A-Z_.\-/]*)?$`),
		googleVcs,
	},
	{
		regexp.MustCompile(`^(github\.com/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]*)?$`),
		githubVcs,
	},
	{
		regexp.MustCompile(`^(bitbucket\.org/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]*)?$`),
		bitbucketVcs,
	},
	{
		regexp.MustCompile(`^(launchpad\.net/([a-z0-9A-Z_.\-]+(/[a-z0-9A-Z_.\-]+)?|~[a-z0-9A-Z_.\-]+/(\+junk|[a-z0-9A-Z_.\-]+)/[a-z0-9A-Z_.\-]+))(/[a-z0-9A-Z_.\-/]+)?$`),
		launchpadVcs,
	},
}

type vcsMatch struct {
	*vcs
	prefix, repo string
}

func googleVcs(repo, path string) (*vcsMatch, os.Error) {
	parts := strings.SplitN(repo, "/", 2)
	url := "https://" + repo
	switch parts[1] {
	case "svn":
		return &vcsMatch{&svn, repo, url}, nil
	case "git":
		return &vcsMatch{&git, repo, url}, nil
	case "hg":
		return &vcsMatch{&hg, repo, url}, nil
	}
	return nil, os.NewError("unsupported googlecode vcs: " + parts[1])
}

func githubVcs(repo, path string) (*vcsMatch, os.Error) {
	if strings.HasSuffix(repo, ".git") {
		return nil, os.NewError("path must not include .git suffix")
	}
	return &vcsMatch{&git, repo, "http://" + repo + ".git"}, nil
}

func bitbucketVcs(repo, path string) (*vcsMatch, os.Error) {
	const bitbucketApiUrl = "https://api.bitbucket.org/1.0/repositories/"

	if strings.HasSuffix(repo, ".git") {
		return nil, os.NewError("path must not include .git suffix")
	}

	parts := strings.SplitN(repo, "/", 2)

	// Ask the bitbucket API what kind of repository this is.
	r, err := http.Get(bitbucketApiUrl + parts[1])
	if err != nil {
		return nil, fmt.Errorf("error querying BitBucket API: %v", err)
	}
	defer r.Body.Close()

	// Did we get a useful response?
	if r.StatusCode != 200 {
		return nil, fmt.Errorf("error querying BitBucket API: %v", r.Status)
	}

	var response struct {
		Vcs string `json:"scm"`
	}
	err = json.NewDecoder(r.Body).Decode(&response)
	if err != nil {
		return nil, fmt.Errorf("error querying BitBucket API: %v", err)
	}

	// Now we should be able to construct a vcsMatch structure
	switch response.Vcs {
	case "git":
		return &vcsMatch{&git, repo, "http://" + repo + ".git"}, nil
	case "hg":
		return &vcsMatch{&hg, repo, "http://" + repo}, nil
	}

	return nil, os.NewError("unsupported bitbucket vcs: " + response.Vcs)
}

func launchpadVcs(repo, path string) (*vcsMatch, os.Error) {
	return &vcsMatch{&bzr, repo, "https://" + repo}, nil
}

// findPublicRepo checks whether pkg is located at one of
// the supported code hosting sites and, if so, returns a match.
func findPublicRepo(pkg string) (*vcsMatch, os.Error) {
	for _, host := range knownHosts {
		if hm := host.pattern.FindStringSubmatch(pkg); hm != nil {
			return host.getVcs(hm[1], hm[2])
		}
	}
	return nil, nil
}

// findAnyRepo looks for a vcs suffix in pkg (.git, etc) and returns a match.
func findAnyRepo(pkg string) (*vcsMatch, os.Error) {
	for _, v := range vcsList {
		i := strings.Index(pkg+"/", v.suffix+"/")
		if i < 0 {
			continue
		}
		if !strings.Contains(pkg[:i], "/") {
			continue // don't match vcs suffix in the host name
		}
		if m := v.find(pkg[:i]); m != nil {
			return m, nil
		}
		return nil, fmt.Errorf("couldn't find %s repository", v.name)
	}
	return nil, nil
}

func (v *vcs) find(pkg string) *vcsMatch {
	for _, proto := range v.protocols {
		for _, suffix := range []string{"", v.suffix} {
			repo := proto + "://" + pkg + suffix
			out, err := exec.Command(v.cmd, v.check, repo).CombinedOutput()
			if err == nil {
				printf("find %s: found %s\n", pkg, repo)
				return &vcsMatch{v, pkg + v.suffix, repo}
			}
			printf("find %s: %s %s %s: %v\n%s\n", pkg, v.cmd, v.check, repo, err, out)
		}
	}
	return nil
}

// isRemote returns true if the first part of the package name looks like a
// hostname - i.e. contains at least one '.' and the last part is at least 2
// characters.
func isRemote(pkg string) bool {
	parts := strings.SplitN(pkg, "/", 2)
	if len(parts) != 2 {
		return false
	}
	parts = strings.Split(parts[0], ".")
	if len(parts) < 2 || len(parts[len(parts)-1]) < 2 {
		return false
	}
	return true
}

// download checks out or updates pkg from the remote server.
func download(pkg, srcDir string) (public bool, err os.Error) {
	if strings.Contains(pkg, "..") {
		err = os.NewError("invalid path (contains ..)")
		return
	}
	m, err := findPublicRepo(pkg)
	if err != nil {
		return
	}
	if m != nil {
		public = true
	} else {
		m, err = findAnyRepo(pkg)
		if err != nil {
			return
		}
	}
	if m == nil {
		err = os.NewError("cannot download: " + pkg)
		return
	}
	err = m.checkoutRepo(srcDir, m.prefix, m.repo)
	return
}

// updateRepo gets a list of tags in the repository and
// checks out the tag closest to the current runtime.Version.
// If no matching tag is found, it just updates to tip.
func (v *vcs) updateRepo(dst string) os.Error {
	if v.tagList == "" || v.tagListRe == nil {
		// TODO(adg): fix for svn
		return run(dst, nil, v.cmd, v.update)
	}

	// Get tag list.
	stderr := new(bytes.Buffer)
	cmd := exec.Command(v.cmd, v.tagList)
	cmd.Dir = dst
	cmd.Stderr = stderr
	b, err := cmd.Output()
	if err != nil {
		errorf("%s %s: %s\n", v.cmd, v.tagList, stderr)
		return err
	}
	var tags []string
	for _, m := range v.tagListRe.FindAllStringSubmatch(string(b), -1) {
		tags = append(tags, m[1])
	}

	// Only use the tag component of runtime.Version.
	ver := strings.Split(runtime.Version(), " ")[0]

	// Select tag.
	if tag := selectTag(ver, tags); tag != "" {
		printf("selecting revision %q\n", tag)
		return run(dst, nil, v.cmd, v.checkout, v.updateRevFlag+tag)
	}

	// No matching tag found, make default selection.
	printf("selecting tip\n")
	return run(dst, nil, v.cmd, v.update)
}

// selectTag returns the closest matching tag for a given version.
// Closest means the latest one that is not after the current release.
// Version "release.rN" matches tags of the form "go.rN" (N being a decimal).
// Version "weekly.YYYY-MM-DD" matches tags like "go.weekly.YYYY-MM-DD".
func selectTag(goVersion string, tags []string) (match string) {
	const rPrefix = "release.r"
	if strings.HasPrefix(goVersion, rPrefix) {
		p := "go.r"
		v, err := strconv.Atof64(goVersion[len(rPrefix):])
		if err != nil {
			return ""
		}
		var matchf float64
		for _, t := range tags {
			if !strings.HasPrefix(t, p) {
				continue
			}
			tf, err := strconv.Atof64(t[len(p):])
			if err != nil {
				continue
			}
			if matchf < tf && tf <= v {
				match, matchf = t, tf
			}
		}
	}
	const wPrefix = "weekly."
	if strings.HasPrefix(goVersion, wPrefix) {
		p := "go.weekly."
		v := goVersion[len(wPrefix):]
		for _, t := range tags {
			if !strings.HasPrefix(t, p) {
				continue
			}
			if match < t && t[len(p):] <= v {
				match = t
			}
		}
	}
	return match
}

// checkoutRepo checks out repo into dst using vcs.
// It tries to check out (or update, if the dst already
// exists and -u was specified on the command line)
// the repository at tag/branch "release".  If there is no
// such tag or branch, it falls back to the repository tip.
func (vcs *vcs) checkoutRepo(srcDir, pkgprefix, repo string) os.Error {
	dst := filepath.Join(srcDir, filepath.FromSlash(pkgprefix))
	dir, err := os.Stat(filepath.Join(dst, vcs.metadir))
	if err == nil && !dir.IsDirectory() {
		return os.NewError("not a directory: " + dst)
	}
	if err != nil {
		parent, _ := filepath.Split(dst)
		if err = os.MkdirAll(parent, 0777); err != nil {
			return err
		}
		if err = run(string(filepath.Separator), nil, vcs.cmd, vcs.clone, repo, dst); err != nil {
			return err
		}
		return vcs.updateRepo(dst)
	}
	if *update {
		// Retrieve new revisions from the remote branch, if the VCS
		// supports this operation independently (e.g. svn doesn't)
		if vcs.pull != "" {
			if vcs.pullForceFlag != "" {
				if err = run(dst, nil, vcs.cmd, vcs.pull, vcs.pullForceFlag); err != nil {
					return err
				}
			} else if err = run(dst, nil, vcs.cmd, vcs.pull); err != nil {
				return err
			}
		}
		// Update to release or latest revision
		return vcs.updateRepo(dst)
	}
	return nil
}
