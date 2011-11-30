// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Download remote packages.

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
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
}

func (v *vcs) String() string {
	return v.name
}

var vcsMap = map[string]*vcs{
	"hg": &vcs{
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
	},

	"git": &vcs{
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
	},

	"svn": &vcs{
		name:      "Subversion",
		cmd:       "svn",
		metadir:   ".svn",
		checkout:  "checkout",
		clone:     "checkout",
		update:    "update",
		check:     "info",
		protocols: []string{"https", "http", "svn"},
		suffix:    ".svn",
	},

	"bzr": &vcs{
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
	},
}

type RemoteRepo interface {
	// IsCheckedOut returns whether this repository is checked
	// out inside the given srcDir (eg, $GOPATH/src).
	IsCheckedOut(srcDir string) bool

	// Repo returns the information about this repository: its url,
	// the part of the import path that forms the repository root,
	// and the version control system it uses. It may discover this
	// information by using the supplied client to make HTTP requests.
	Repo(*http.Client) (url, root string, vcs *vcs, err error)
}

type host struct {
	pattern *regexp.Regexp
	repo    func(repo string) (RemoteRepo, error)
}

var knownHosts = []host{
	{
		regexp.MustCompile(`^([a-z0-9\-]+\.googlecode\.com/(svn|git|hg))(/[a-z0-9A-Z_.\-/]+)?$`),
		matchGoogleRepo,
	},
	{
		regexp.MustCompile(`^code\.google\.com/p/([a-z0-9\-]+\.[a-z0-9\-]+)(/[a-z0-9A-Z_.\-/]+)?$`),
		matchGoogleSubrepo,
	},
	{
		regexp.MustCompile(`^(github\.com/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]+)?$`),
		matchGithubRepo,
	},
	{
		regexp.MustCompile(`^(bitbucket\.org/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]+)?$`),
		matchBitbucketRepo,
	},
	{
		regexp.MustCompile(`^(launchpad\.net/([a-z0-9A-Z_.\-]+(/[a-z0-9A-Z_.\-]+)?|~[a-z0-9A-Z_.\-]+/(\+junk|[a-z0-9A-Z_.\-]+)/[a-z0-9A-Z_.\-]+))(/[a-z0-9A-Z_.\-/]+)?$`),
		matchLaunchpadRepo,
	},
}

// baseRepo is the base implementation of RemoteRepo.
type baseRepo struct {
	url, root string
	vcs       *vcs
}

func (r *baseRepo) Repo(*http.Client) (url, root string, vcs *vcs, err error) {
	return r.url, r.root, r.vcs, nil
}

// IsCheckedOut reports whether the repo root inside srcDir contains a
// repository metadir. It updates the baseRepo's vcs field if necessary.
func (r *baseRepo) IsCheckedOut(srcDir string) bool {
	pkgPath := filepath.Join(srcDir, r.root)
	if r.vcs == nil {
		for _, vcs := range vcsMap {
			if isDir(filepath.Join(pkgPath, vcs.metadir)) {
				r.vcs = vcs
				return true
			}
		}
		return false
	}
	return isDir(filepath.Join(pkgPath, r.vcs.metadir))
}

// matchGoogleRepo handles matches of the form "repo.googlecode.com/vcs/path".
func matchGoogleRepo(root string) (RemoteRepo, error) {
	p := strings.SplitN(root, "/", 2)
	if vcs := vcsMap[p[1]]; vcs != nil {
		return &baseRepo{"https://" + root, root, vcs}, nil
	}
	return nil, errors.New("unsupported googlecode vcs: " + p[1])
}

// matchGithubRepo handles matches for github.com repositories.
func matchGithubRepo(root string) (RemoteRepo, error) {
	if strings.HasSuffix(root, ".git") {
		return nil, errors.New("path must not include .git suffix")
	}
	return &baseRepo{"http://" + root + ".git", root, vcsMap["git"]}, nil
}

// matchLaunchpadRepo handles matches for launchpad.net repositories.
func matchLaunchpadRepo(root string) (RemoteRepo, error) {
	return &baseRepo{"https://" + root, root, vcsMap["bzr"]}, nil
}

// matchGoogleSubrepo matches repos like "code.google.com/p/repo.subrepo/path".
// Note that it doesn't match primary Google Code repositories,
// which should use the "foo.googlecode.com" form only. (for now)
func matchGoogleSubrepo(id string) (RemoteRepo, error) {
	root := "code.google.com/p/" + id
	return &googleSubrepo{baseRepo{"https://" + root, root, nil}}, nil
}

// googleSubrepo implements a RemoteRepo that discovers a Google Code
// repository's VCS type by scraping the code.google.com source checkout page.
type googleSubrepo struct{ baseRepo }

var googleSubrepoRe = regexp.MustCompile(`id="checkoutcmd">(hg|git|svn)`)

func (r *googleSubrepo) Repo(client *http.Client) (url, root string, vcs *vcs, err error) {
	if r.vcs != nil {
		return r.url, r.root, r.vcs, nil
	}

	// Use the code.google.com source checkout page to find the VCS type.
	const prefix = "code.google.com/p/"
	p := strings.SplitN(r.root[len(prefix):], ".", 2)
	u := fmt.Sprintf("https://%s%s/source/checkout?repo=%s", prefix, p[0], p[1])
	resp, err := client.Get(u)
	if err != nil {
		return "", "", nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", "", nil, fmt.Errorf("fetching %s: %v", u, resp.Status)
	}
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", "", nil, fmt.Errorf("fetching %s: %v", u, err)
	}

	// Scrape result for vcs details.
	m := googleSubrepoRe.FindSubmatch(b)
	if len(m) == 2 {
		if v := vcsMap[string(m[1])]; v != nil {
			r.vcs = v
			return r.url, r.root, r.vcs, nil
		}
	}

	return "", "", nil, errors.New("could not detect googlecode vcs")
}

// matchBitbucketRepo handles matches for all bitbucket.org repositories.
func matchBitbucketRepo(root string) (RemoteRepo, error) {
	if strings.HasSuffix(root, ".git") {
		return nil, errors.New("path must not include .git suffix")
	}
	return &bitbucketRepo{baseRepo{root: root}}, nil
}

// bitbucketRepo implements a RemoteRepo that uses the BitBucket API to
// discover the repository's VCS type.
type bitbucketRepo struct{ baseRepo }

func (r *bitbucketRepo) Repo(client *http.Client) (url, root string, vcs *vcs, err error) {
	if r.vcs != nil && r.url != "" {
		return r.url, r.root, r.vcs, nil
	}

	// Use the BitBucket API to find which kind of repository this is.
	const apiUrl = "https://api.bitbucket.org/1.0/repositories/"
	resp, err := client.Get(apiUrl + strings.SplitN(r.root, "/", 2)[1])
	if err != nil {
		return "", "", nil, fmt.Errorf("BitBucket API: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", "", nil, fmt.Errorf("BitBucket API: %v", resp.Status)
	}
	var response struct {
		Vcs string `json:"scm"`
	}
	err = json.NewDecoder(resp.Body).Decode(&response)
	if err != nil {
		return "", "", nil, fmt.Errorf("BitBucket API: %v", err)
	}
	switch response.Vcs {
	case "git":
		r.url = "http://" + r.root + ".git"
	case "hg":
		r.url = "http://" + r.root
	default:
		return "", "", nil, errors.New("unsupported bitbucket vcs: " + response.Vcs)
	}
	if r.vcs = vcsMap[response.Vcs]; r.vcs == nil {
		panic("vcs is nil when it should not be")
	}
	return r.url, r.root, r.vcs, nil
}

// findPublicRepo checks whether importPath is a well-formed path for one of
// the supported code hosting sites and, if so, returns a RemoteRepo.
func findPublicRepo(importPath string) (RemoteRepo, error) {
	for _, host := range knownHosts {
		if hm := host.pattern.FindStringSubmatch(importPath); hm != nil {
			return host.repo(hm[1])
		}
	}
	return nil, nil
}

// findAnyRepo matches import paths with a repo suffix (.git, etc).
func findAnyRepo(importPath string) RemoteRepo {
	for _, v := range vcsMap {
		i := strings.Index(importPath+"/", v.suffix+"/")
		if i < 0 {
			continue
		}
		if !strings.Contains(importPath[:i], "/") {
			continue // don't match vcs suffix in the host name
		}
		return &anyRepo{
			baseRepo{
				root: importPath[:i] + v.suffix,
				vcs:  v,
			},
			importPath[:i],
		}
	}
	return nil
}

// anyRepo implements an discoverable remote repo with a suffix (.git, etc).
type anyRepo struct {
	baseRepo
	rootWithoutSuffix string
}

func (r *anyRepo) Repo(*http.Client) (url, root string, vcs *vcs, err error) {
	if r.url != "" {
		return r.url, r.root, r.vcs, nil
	}
	url, err = r.vcs.findURL(r.rootWithoutSuffix)
	if url == "" && err == nil {
		err = fmt.Errorf("couldn't find %s repository", r.vcs.name)
	}
	if err != nil {
		return "", "", nil, err
	}
	r.url = url
	return r.url, r.root, r.vcs, nil
}

// findURL finds the URL for a given repo root by trying each combination of
// protocol and suffix in series.
func (v *vcs) findURL(root string) (string, error) {
	for _, proto := range v.protocols {
		for _, suffix := range []string{"", v.suffix} {
			url := proto + "://" + root + suffix
			out, err := exec.Command(v.cmd, v.check, url).CombinedOutput()
			if err == nil {
				printf("find %s: found %s\n", root, url)
				return url, nil
			}
			printf("findURL(%s): %s %s %s: %v\n%s\n", root, v.cmd, v.check, url, err, out)
		}
	}
	return "", nil
}

// download checks out or updates the specified package from the remote server.
func download(importPath, srcDir string) (public bool, err error) {
	if strings.Contains(importPath, "..") {
		err = errors.New("invalid path (contains ..)")
		return
	}

	repo, err := findPublicRepo(importPath)
	if err != nil {
		return false, err
	}
	if repo != nil {
		public = true
	} else {
		repo = findAnyRepo(importPath)
	}
	if repo == nil {
		err = errors.New("cannot download: " + importPath)
		return
	}
	err = checkoutRepo(srcDir, repo)
	return
}

// checkoutRepo checks out repo into srcDir (if it's not checked out already)
// and, if the -u flag is set, updates the repository.
func checkoutRepo(srcDir string, repo RemoteRepo) error {
	if !repo.IsCheckedOut(srcDir) {
		// do checkout
		url, root, vcs, err := repo.Repo(http.DefaultClient)
		if err != nil {
			return err
		}
		repoPath := filepath.Join(srcDir, root)
		parent, _ := filepath.Split(repoPath)
		if err = os.MkdirAll(parent, 0777); err != nil {
			return err
		}
		if err = run(string(filepath.Separator), nil, vcs.cmd, vcs.clone, url, repoPath); err != nil {
			return err
		}
		return vcs.updateRepo(repoPath)
	}
	if *update {
		// do update
		_, root, vcs, err := repo.Repo(http.DefaultClient)
		if err != nil {
			return err
		}
		repoPath := filepath.Join(srcDir, root)
		// Retrieve new revisions from the remote branch, if the VCS
		// supports this operation independently (e.g. svn doesn't)
		if vcs.pull != "" {
			if vcs.pullForceFlag != "" {
				if err = run(repoPath, nil, vcs.cmd, vcs.pull, vcs.pullForceFlag); err != nil {
					return err
				}
			} else if err = run(repoPath, nil, vcs.cmd, vcs.pull); err != nil {
				return err
			}
		}
		// Update to release or latest revision
		return vcs.updateRepo(repoPath)
	}
	return nil
}

// updateRepo gets a list of tags in the repository and
// checks out the tag closest to the current runtime.Version.
// If no matching tag is found, it just updates to tip.
func (v *vcs) updateRepo(repoPath string) error {
	if v.tagList == "" || v.tagListRe == nil {
		// TODO(adg): fix for svn
		return run(repoPath, nil, v.cmd, v.update)
	}

	// Get tag list.
	stderr := new(bytes.Buffer)
	cmd := exec.Command(v.cmd, v.tagList)
	cmd.Dir = repoPath
	cmd.Stderr = stderr
	out, err := cmd.Output()
	if err != nil {
		return &RunError{strings.Join(cmd.Args, " "), repoPath, out, err}
	}
	var tags []string
	for _, m := range v.tagListRe.FindAllStringSubmatch(string(out), -1) {
		tags = append(tags, m[1])
	}

	// Only use the tag component of runtime.Version.
	ver := strings.Split(runtime.Version(), " ")[0]

	// Select tag.
	if tag := selectTag(ver, tags); tag != "" {
		printf("selecting revision %q\n", tag)
		return run(repoPath, nil, v.cmd, v.checkout, v.updateRevFlag+tag)
	}

	// No matching tag found, make default selection.
	printf("selecting tip\n")
	return run(repoPath, nil, v.cmd, v.update)
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

func isDir(dir string) bool {
	fi, err := os.Stat(dir)
	return err == nil && fi.IsDir()
}
