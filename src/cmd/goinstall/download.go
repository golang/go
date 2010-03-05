// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Download remote packages.

package main

import (
	"http"
	"os"
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

// download checks out or updates pkg from the remote server.
func download(pkg string) (string, os.Error) {
	if strings.Index(pkg, "..") >= 0 {
		return "", os.ErrorString("invalid path (contains ..)")
	}
	if m := bitbucket.MatchStrings(pkg); m != nil {
		if err := vcsCheckout(&hg, root+m[1], "http://"+m[1], m[1]); err != nil {
			return "", err
		}
		return root + pkg, nil
	}
	if m := googlecode.MatchStrings(pkg); m != nil {
		var v *vcs
		switch m[2] {
		case "hg":
			v = &hg
		case "svn":
			v = &svn
		default:
			// regexp only allows hg, svn to get through
			panic("missing case in download: ", pkg)
		}
		if err := vcsCheckout(v, root+m[1], "http://"+m[1], m[1]); err != nil {
			return "", err
		}
		return root + pkg, nil
	}
	if m := github.MatchStrings(pkg); m != nil {
		if strings.HasSuffix(m[1], ".git") {
			return "", os.ErrorString("repository " + pkg + " should not have .git suffix")
		}
		if err := vcsCheckout(&git, root+m[1], "http://"+m[1]+".git", m[1]); err != nil {
			return "", err
		}
		return root + pkg, nil
	}
	return "", os.ErrorString("unknown repository: " + pkg)
}

// a vcs represents a version control system
// like Mercurial, Git, or Subversion.
type vcs struct {
	cmd            string
	metadir        string
	clone          string
	update         string
	pull           string
	log            string
	logLimitFlag   string
	logReleaseFlag string
}

var hg = vcs{
	cmd:            "hg",
	metadir:        ".hg",
	clone:          "clone",
	update:         "update",
	pull:           "pull",
	log:            "log",
	logLimitFlag:   "-l1",
	logReleaseFlag: "-rrelease",
}

var git = vcs{
	cmd:            "git",
	metadir:        ".git",
	clone:          "clone",
	update:         "checkout",
	pull:           "fetch",
	log:            "log",
	logLimitFlag:   "-n1",
	logReleaseFlag: "release",
}

var svn = vcs{
	cmd:            "svn",
	metadir:        ".svn",
	clone:          "checkout",
	update:         "update",
	pull:           "",
	log:            "log",
	logLimitFlag:   "-l1",
	logReleaseFlag: "release",
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
		if err := os.MkdirAll(dst, 0777); err != nil {
			return err
		}
		if err := run("/", nil, vcs.cmd, vcs.clone, repo, dst); err != nil {
			return err
		}
		quietRun(dst, nil, vcs.cmd, vcs.update, "release")

		// success on first installation - report
		maybeReportToDashboard(dashpath)
	} else if *update {
		if vcs.pull != "" {
			if err := run(dst, nil, vcs.cmd, vcs.pull); err != nil {
				return err
			}
		}
		// check for release with hg log -l 1 -r release
		// if success, hg update release
		// else hg update
		if err := quietRun(dst, nil, vcs.cmd, vcs.log, vcs.logLimitFlag, vcs.logReleaseFlag); err == nil {
			if err := run(dst, nil, vcs.cmd, vcs.update, "release"); err != nil {
				return err
			}
		} else {
			if err := run(dst, nil, vcs.cmd, vcs.update); err != nil {
				return err
			}
		}
	}
	return nil
}
