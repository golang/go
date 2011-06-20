// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Download remote packages.

package main

import (
	"exec"
	"http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
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

type host struct {
	pattern  *regexp.Regexp
	protocol string
}

// a vcs represents a version control system
// like Mercurial, Git, or Subversion.
type vcs struct {
	name              string
	cmd               string
	metadir           string
	checkout          string
	clone             string
	update            string
	updateReleaseFlag string
	pull              string
	pullForceFlag     string
	log               string
	logLimitFlag      string
	logReleaseFlag    string
	check             string
	protocols         []string
	suffix            string
	tryPrefixes       bool
	defaultHosts      []host

	// Is this tool present? (set by findTools)
	available bool
}

type vcsMatch struct {
	*vcs
	prefix, repo string
}

var hg = vcs{
	name:              "Mercurial",
	cmd:               "hg",
	metadir:           ".hg",
	checkout:          "checkout",
	clone:             "clone",
	update:            "update",
	updateReleaseFlag: "release",
	pull:              "pull",
	log:               "log",
	logLimitFlag:      "-l1",
	logReleaseFlag:    "-rrelease",
	check:             "identify",
	protocols:         []string{"http"},
	tryPrefixes:       true,
	defaultHosts: []host{
		{regexp.MustCompile(`^([a-z0-9\-]+\.googlecode\.com/hg)(/[a-z0-9A-Z_.\-/]*)?$`), "https"},
		{regexp.MustCompile(`^(bitbucket\.org/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]*)?$`), "http"},
	},
}

var git = vcs{
	name:              "Git",
	cmd:               "git",
	metadir:           ".git",
	checkout:          "checkout",
	clone:             "clone",
	update:            "pull",
	updateReleaseFlag: "release",
	pull:              "fetch",
	log:               "show-ref",
	logLimitFlag:      "",
	logReleaseFlag:    "release",
	check:             "peek-remote",
	protocols:         []string{"git", "http"},
	suffix:            ".git",
	tryPrefixes:       true,
	defaultHosts: []host{
		{regexp.MustCompile(`^(github\.com/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]*)?$`), "http"},
	},
}

var svn = vcs{
	name:              "Subversion",
	cmd:               "svn",
	metadir:           ".svn",
	checkout:          "checkout",
	clone:             "checkout",
	update:            "update",
	updateReleaseFlag: "release",
	log:               "log",
	logLimitFlag:      "-l1",
	logReleaseFlag:    "release",
	check:             "info",
	protocols:         []string{"http", "svn"},
	tryPrefixes:       false,
	defaultHosts: []host{
		{regexp.MustCompile(`^([a-z0-9\-]+\.googlecode\.com/svn)(/[a-z0-9A-Z_.\-/]*)?$`), "https"},
	},
}

var bzr = vcs{
	name:              "Bazaar",
	cmd:               "bzr",
	metadir:           ".bzr",
	checkout:          "update",
	clone:             "branch",
	update:            "update",
	updateReleaseFlag: "-rrelease",
	pull:              "pull",
	pullForceFlag:     "--overwrite",
	log:               "log",
	logLimitFlag:      "-l1",
	logReleaseFlag:    "-rrelease",
	check:             "info",
	protocols:         []string{"http", "bzr"},
	tryPrefixes:       true,
	defaultHosts: []host{
		{regexp.MustCompile(`^(launchpad\.net/([a-z0-9A-Z_.\-]+(/[a-z0-9A-Z_.\-]+)?|~[a-z0-9A-Z_.\-]+/(\+junk|[a-z0-9A-Z_.\-]+)/[a-z0-9A-Z_.\-]+))(/[a-z0-9A-Z_.\-/]+)?$`), "https"},
	},
}

var vcsList = []*vcs{&git, &hg, &bzr, &svn}

func potentialPrefixes(pkg string) (prefixes []string) {
	parts := strings.Split(pkg, "/", -1)
	elem := parts[0]
	for _, part := range parts[1:] {
		elem = path.Join(elem, part)
		prefixes = append(prefixes, elem)
	}
	return
}

func tryCommand(c chan *vcsMatch, v *vcs, prefixes []string) {
	// try empty suffix and v.suffix if non-empty
	suffixes := []string{""}
	if v.suffix != "" {
		suffixes = append(suffixes, v.suffix)
	}
	for _, proto := range v.protocols {
		for _, prefix := range prefixes {
			for _, suffix := range suffixes {
				repo := proto + "://" + prefix + suffix
				vlogf("try: %s %s %s\n", v.cmd, v.check, repo)
				if exec.Command(v.cmd, v.check, repo).Run() == nil {
					c <- &vcsMatch{v, prefix, repo}
					return
				}
			}
		}
	}
	c <- nil
}

var findToolsOnce sync.Once

func findTools() {
	for _, v := range vcsList {
		v.available = exec.Command(v.cmd, "help").Run() == nil
	}
}

var logMissingToolsOnce sync.Once

func logMissingTools() {
	for _, v := range vcsList {
		if !v.available {
			logf("%s not found; %s packages will be ignored\n", v.cmd, v.name)
		}
	}
}

func findVcs(pkg string) *vcsMatch {
	findToolsOnce.Do(findTools)

	// we don't know how much of the name constitutes the repository prefix
	// so build a list of possibilities
	prefixes := potentialPrefixes(pkg)

	c := make(chan *vcsMatch, len(vcsList))
	for _, v := range vcsList {
		if !v.available {
			c <- nil
			continue
		}
		if v.tryPrefixes {
			go tryCommand(c, v, prefixes)
		} else {
			go tryCommand(c, v, []string{pkg})
		}
	}
	for _ = range vcsList {
		if m := <-c; m != nil {
			return m
		}
	}

	logMissingToolsOnce.Do(logMissingTools)
	return nil
}

// isRemote returns true if the first part of the package name looks like a
// hostname - i.e. contains at least one '.' and the last part is at least 2
// characters.
func isRemote(pkg string) bool {
	parts := strings.Split(pkg, "/", 2)
	if len(parts) != 2 {
		return false
	}
	parts = strings.Split(parts[0], ".", -1)
	if len(parts) < 2 || len(parts[len(parts)-1]) < 2 {
		return false
	}
	return true
}

// download checks out or updates pkg from the remote server.
func download(pkg, srcDir string) os.Error {
	if strings.Contains(pkg, "..") {
		return os.ErrorString("invalid path (contains ..)")
	}
	var m *vcsMatch
	for _, v := range vcsList {
		for _, host := range v.defaultHosts {
			if hm := host.pattern.FindStringSubmatch(pkg); hm != nil {
				if v.suffix != "" && strings.HasSuffix(hm[1], v.suffix) {
					return os.ErrorString("repository " + pkg + " should not have " + v.suffix + " suffix")
				}
				repo := host.protocol + "://" + hm[1] + v.suffix
				m = &vcsMatch{v, hm[1], repo}
			}
		}
	}
	if m == nil {
		m = findVcs(pkg)
	}
	if m == nil {
		return os.ErrorString("cannot download: " + pkg)
	}
	return vcsCheckout(m.vcs, srcDir, m.prefix, m.repo, pkg)
}

// Try to detect if a "release" tag exists.  If it does, update
// to the tagged version, otherwise just update the current branch.
// NOTE(_nil): svn will always fail because it is trying to get
// the revision history of a file named "release" instead of
// looking for a commit with a release tag
func (v *vcs) updateRepo(dst string) os.Error {
	if err := quietRun(dst, nil, v.cmd, v.log, v.logLimitFlag, v.logReleaseFlag); err == nil {
		if err := run(dst, nil, v.cmd, v.checkout, v.updateReleaseFlag); err != nil {
			return err
		}
	} else if err := run(dst, nil, v.cmd, v.update); err != nil {
		return err
	}
	return nil
}

// vcsCheckout checks out repo into dst using vcs.
// It tries to check out (or update, if the dst already
// exists and -u was specified on the command line)
// the repository at tag/branch "release".  If there is no
// such tag or branch, it falls back to the repository tip.
func vcsCheckout(vcs *vcs, srcDir, pkgprefix, repo, dashpath string) os.Error {
	dst := filepath.Join(srcDir, filepath.FromSlash(pkgprefix))
	dir, err := os.Stat(filepath.Join(dst, vcs.metadir))
	if err == nil && !dir.IsDirectory() {
		return os.ErrorString("not a directory: " + dst)
	}
	if err != nil {
		parent, _ := filepath.Split(dst)
		if err := os.MkdirAll(parent, 0777); err != nil {
			return err
		}
		if err := run(string(filepath.Separator), nil, vcs.cmd, vcs.clone, repo, dst); err != nil {
			return err
		}
		if err := vcs.updateRepo(dst); err != nil {
			return err
		}
		// success on first installation - report
		maybeReportToDashboard(dashpath)
	} else if *update {
		// Retrieve new revisions from the remote branch, if the VCS
		// supports this operation independently (e.g. svn doesn't)
		if vcs.pull != "" {
			if vcs.pullForceFlag != "" {
				if err := run(dst, nil, vcs.cmd, vcs.pull, vcs.pullForceFlag); err != nil {
					return err
				}
			} else if err := run(dst, nil, vcs.cmd, vcs.pull); err != nil {
				return err
			}
		}

		// Update to release or latest revision
		if err := vcs.updateRepo(dst); err != nil {
			return err
		}
	}
	return nil
}
