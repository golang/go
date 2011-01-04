// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Download remote packages.

package main

import (
	"http"
	"os"
	"path"
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

var googlecode = regexp.MustCompile(`^([a-z0-9\-]+\.googlecode\.com/(svn|hg))(/[a-z0-9A-Z_.\-/]*)?$`)
var github = regexp.MustCompile(`^(github\.com/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]*)?$`)
var bitbucket = regexp.MustCompile(`^(bitbucket\.org/[a-z0-9A-Z_.\-]+/[a-z0-9A-Z_.\-]+)(/[a-z0-9A-Z_.\-/]*)?$`)
var launchpad = regexp.MustCompile(`^(launchpad\.net/([a-z0-9A-Z_.\-]+(/[a-z0-9A-Z_.\-]+)?|~[a-z0-9A-Z_.\-]+/(\+junk|[a-z0-9A-Z_.\-]+)/[a-z0-9A-Z_.\-]+))(/[a-z0-9A-Z_.\-/]+)?$`)

// download checks out or updates pkg from the remote server.
func download(pkg string) (string, os.Error) {
	if strings.Contains(pkg, "..") {
		return "", os.ErrorString("invalid path (contains ..)")
	}
	if m := bitbucket.FindStringSubmatch(pkg); m != nil {
		if err := vcsCheckout(&hg, root+m[1], "http://"+m[1], m[1]); err != nil {
			return "", err
		}
		return root + pkg, nil
	}
	if m := googlecode.FindStringSubmatch(pkg); m != nil {
		var v *vcs
		switch m[2] {
		case "hg":
			v = &hg
		case "svn":
			v = &svn
		default:
			// regexp only allows hg, svn to get through
			panic("missing case in download: " + pkg)
		}
		if err := vcsCheckout(v, root+m[1], "https://"+m[1], m[1]); err != nil {
			return "", err
		}
		return root + pkg, nil
	}
	if m := github.FindStringSubmatch(pkg); m != nil {
		if strings.HasSuffix(m[1], ".git") {
			return "", os.ErrorString("repository " + pkg + " should not have .git suffix")
		}
		if err := vcsCheckout(&git, root+m[1], "http://"+m[1]+".git", m[1]); err != nil {
			return "", err
		}
		return root + pkg, nil
	}
	if m := launchpad.FindStringSubmatch(pkg); m != nil {
		// Either lp.net/<project>[/<series>[/<path>]]
		//	 or lp.net/~<user or team>/<project>/<branch>[/<path>]
		if err := vcsCheckout(&bzr, root+m[1], "https://"+m[1], m[1]); err != nil {
			return "", err
		}
		return root + pkg, nil
	}
	return "", os.ErrorString("unknown repository: " + pkg)
}

// a vcs represents a version control system
// like Mercurial, Git, or Subversion.
type vcs struct {
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
}

var hg = vcs{
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
}

var git = vcs{
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
}

var svn = vcs{
	cmd:               "svn",
	metadir:           ".svn",
	checkout:          "checkout",
	clone:             "checkout",
	update:            "update",
	updateReleaseFlag: "release",
	log:               "log",
	logLimitFlag:      "-l1",
	logReleaseFlag:    "release",
}

var bzr = vcs{
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
func vcsCheckout(vcs *vcs, dst, repo, dashpath string) os.Error {
	dir, err := os.Stat(dst + "/" + vcs.metadir)
	if err == nil && !dir.IsDirectory() {
		return os.ErrorString("not a directory: " + dst)
	}
	if err != nil {
		parent, _ := path.Split(dst)
		if err := os.MkdirAll(parent, 0777); err != nil {
			return err
		}
		if err := run("/", nil, vcs.cmd, vcs.clone, repo, dst); err != nil {
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
