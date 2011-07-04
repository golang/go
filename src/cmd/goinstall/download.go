// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Download remote packages.

package main

import (
	"exec"
	"fmt"
	"http"
	"os"
	"path/filepath"
	"regexp"
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
	defaultHosts      []host
}

type host struct {
	pattern  *regexp.Regexp
	protocol string
	suffix   string
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
	protocols:         []string{"https", "http"},
	suffix:            ".hg",
	defaultHosts: []host{
		{regexp.MustCompile(`^([a-z0-9\-]+\.googlecode\.com/hg)(/[a-z0-9A-Z_.\-/]*)?$`), "https", ""},
		{regexp.MustCompile(`^(bitbucket\.org/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]*)?$`), "http", ""},
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
	check:             "ls-remote",
	protocols:         []string{"git", "https", "http"},
	suffix:            ".git",
	defaultHosts: []host{
		{regexp.MustCompile(`^(github\.com/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]*)?$`), "http", ".git"},
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
	protocols:         []string{"https", "http", "svn"},
	suffix:            ".svn",
	defaultHosts: []host{
		{regexp.MustCompile(`^([a-z0-9\-]+\.googlecode\.com/svn)(/[a-z0-9A-Z_.\-/]*)?$`), "https", ""},
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
	protocols:         []string{"https", "http", "bzr"},
	suffix:            ".bzr",
	defaultHosts: []host{
		{regexp.MustCompile(`^(launchpad\.net/([a-z0-9A-Z_.\-]+(/[a-z0-9A-Z_.\-]+)?|~[a-z0-9A-Z_.\-]+/(\+junk|[a-z0-9A-Z_.\-]+)/[a-z0-9A-Z_.\-]+))(/[a-z0-9A-Z_.\-/]+)?$`), "https", ""},
	},
}

var vcsList = []*vcs{&git, &hg, &bzr, &svn}

type vcsMatch struct {
	*vcs
	prefix, repo string
}

// findHostedRepo checks whether pkg is located at one of
// the supported code hosting sites and, if so, returns a match.
func findHostedRepo(pkg string) (*vcsMatch, os.Error) {
	for _, v := range vcsList {
		for _, host := range v.defaultHosts {
			if hm := host.pattern.FindStringSubmatch(pkg); hm != nil {
				if host.suffix != "" && strings.HasSuffix(hm[1], host.suffix) {
					return nil, os.NewError("repository " + pkg + " should not have " + v.suffix + " suffix")
				}
				repo := host.protocol + "://" + hm[1] + host.suffix
				return &vcsMatch{v, hm[1], repo}, nil
			}
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
func download(pkg, srcDir string) (dashReport bool, err os.Error) {
	if strings.Contains(pkg, "..") {
		err = os.NewError("invalid path (contains ..)")
		return
	}
	m, err := findHostedRepo(pkg)
	if err != nil {
		return
	}
	if m != nil {
		dashReport = true // only report public code hosting sites
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
	installed, err := m.checkoutRepo(srcDir, m.prefix, m.repo)
	if err != nil {
		return
	}
	if !installed {
		dashReport = false
	}
	return
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

// checkoutRepo checks out repo into dst using vcs.
// It tries to check out (or update, if the dst already
// exists and -u was specified on the command line)
// the repository at tag/branch "release".  If there is no
// such tag or branch, it falls back to the repository tip.
func (vcs *vcs) checkoutRepo(srcDir, pkgprefix, repo string) (installed bool, err os.Error) {
	dst := filepath.Join(srcDir, filepath.FromSlash(pkgprefix))
	dir, err := os.Stat(filepath.Join(dst, vcs.metadir))
	if err == nil && !dir.IsDirectory() {
		err = os.NewError("not a directory: " + dst)
		return
	}
	if err != nil {
		parent, _ := filepath.Split(dst)
		if err = os.MkdirAll(parent, 0777); err != nil {
			return
		}
		if err = run(string(filepath.Separator), nil, vcs.cmd, vcs.clone, repo, dst); err != nil {
			return
		}
		if err = vcs.updateRepo(dst); err != nil {
			return
		}
		installed = true
	} else if *update {
		// Retrieve new revisions from the remote branch, if the VCS
		// supports this operation independently (e.g. svn doesn't)
		if vcs.pull != "" {
			if vcs.pullForceFlag != "" {
				if err = run(dst, nil, vcs.cmd, vcs.pull, vcs.pullForceFlag); err != nil {
					return
				}
			} else if err = run(dst, nil, vcs.cmd, vcs.pull); err != nil {
				return
			}
		}
		// Update to release or latest revision
		if err = vcs.updateRepo(dst); err != nil {
			return
		}
	}
	return
}
