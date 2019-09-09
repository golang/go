// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"encoding/xml"
	"fmt"
	"internal/lazyregexp"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/lockedfile"
	"cmd/go/internal/par"
	"cmd/go/internal/str"
)

// A VCSError indicates an error using a version control system.
// The implication of a VCSError is that we know definitively where
// to get the code, but we can't access it due to the error.
// The caller should report this error instead of continuing to probe
// other possible module paths.
//
// TODO(golang.org/issue/31730): See if we can invert this. (Return a
// distinguished error for “repo not found” and treat everything else
// as terminal.)
type VCSError struct {
	Err error
}

func (e *VCSError) Error() string { return e.Err.Error() }

func vcsErrorf(format string, a ...interface{}) error {
	return &VCSError{Err: fmt.Errorf(format, a...)}
}

func NewRepo(vcs, remote string) (Repo, error) {
	type key struct {
		vcs    string
		remote string
	}
	type cached struct {
		repo Repo
		err  error
	}
	c := vcsRepoCache.Do(key{vcs, remote}, func() interface{} {
		repo, err := newVCSRepo(vcs, remote)
		if err != nil {
			err = &VCSError{err}
		}
		return cached{repo, err}
	}).(cached)

	return c.repo, c.err
}

var vcsRepoCache par.Cache

type vcsRepo struct {
	mu lockedfile.Mutex // protects all commands, so we don't have to decide which are safe on a per-VCS basis

	remote string
	cmd    *vcsCmd
	dir    string

	tagsOnce sync.Once
	tags     map[string]bool

	branchesOnce sync.Once
	branches     map[string]bool

	fetchOnce sync.Once
	fetchErr  error
}

func newVCSRepo(vcs, remote string) (Repo, error) {
	if vcs == "git" {
		return newGitRepo(remote, false)
	}
	cmd := vcsCmds[vcs]
	if cmd == nil {
		return nil, fmt.Errorf("unknown vcs: %s %s", vcs, remote)
	}
	if !strings.Contains(remote, "://") {
		return nil, fmt.Errorf("invalid vcs remote: %s %s", vcs, remote)
	}

	r := &vcsRepo{remote: remote, cmd: cmd}
	var err error
	r.dir, r.mu.Path, err = WorkDir(vcsWorkDirType+vcs, r.remote)
	if err != nil {
		return nil, err
	}

	if cmd.init == nil {
		return r, nil
	}

	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	if _, err := os.Stat(filepath.Join(r.dir, "."+vcs)); err != nil {
		if _, err := Run(r.dir, cmd.init(r.remote)); err != nil {
			os.RemoveAll(r.dir)
			return nil, err
		}
	}
	return r, nil
}

const vcsWorkDirType = "vcs1."

type vcsCmd struct {
	vcs           string                                            // vcs name "hg"
	init          func(remote string) []string                      // cmd to init repo to track remote
	tags          func(remote string) []string                      // cmd to list local tags
	tagRE         *lazyregexp.Regexp                                // regexp to extract tag names from output of tags cmd
	branches      func(remote string) []string                      // cmd to list local branches
	branchRE      *lazyregexp.Regexp                                // regexp to extract branch names from output of tags cmd
	badLocalRevRE *lazyregexp.Regexp                                // regexp of names that must not be served out of local cache without doing fetch first
	statLocal     func(rev, remote string) []string                 // cmd to stat local rev
	parseStat     func(rev, out string) (*RevInfo, error)           // cmd to parse output of statLocal
	fetch         []string                                          // cmd to fetch everything from remote
	latest        string                                            // name of latest commit on remote (tip, HEAD, etc)
	readFile      func(rev, file, remote string) []string           // cmd to read rev's file
	readZip       func(rev, subdir, remote, target string) []string // cmd to read rev's subdir as zip file
}

var re = lazyregexp.New

var vcsCmds = map[string]*vcsCmd{
	"hg": {
		vcs: "hg",
		init: func(remote string) []string {
			return []string{"hg", "clone", "-U", "--", remote, "."}
		},
		tags: func(remote string) []string {
			return []string{"hg", "tags", "-q"}
		},
		tagRE: re(`(?m)^[^\n]+$`),
		branches: func(remote string) []string {
			return []string{"hg", "branches", "-c", "-q"}
		},
		branchRE:      re(`(?m)^[^\n]+$`),
		badLocalRevRE: re(`(?m)^(tip)$`),
		statLocal: func(rev, remote string) []string {
			return []string{"hg", "log", "-l1", "-r", rev, "--template", "{node} {date|hgdate} {tags}"}
		},
		parseStat: hgParseStat,
		fetch:     []string{"hg", "pull", "-f"},
		latest:    "tip",
		readFile: func(rev, file, remote string) []string {
			return []string{"hg", "cat", "-r", rev, file}
		},
		readZip: func(rev, subdir, remote, target string) []string {
			pattern := []string{}
			if subdir != "" {
				pattern = []string{"-I", subdir + "/**"}
			}
			return str.StringList("hg", "archive", "-t", "zip", "--no-decode", "-r", rev, "--prefix=prefix/", pattern, "--", target)
		},
	},

	"svn": {
		vcs:  "svn",
		init: nil, // no local checkout
		tags: func(remote string) []string {
			return []string{"svn", "list", "--", strings.TrimSuffix(remote, "/trunk") + "/tags"}
		},
		tagRE: re(`(?m)^(.*?)/?$`),
		statLocal: func(rev, remote string) []string {
			suffix := "@" + rev
			if rev == "latest" {
				suffix = ""
			}
			return []string{"svn", "log", "-l1", "--xml", "--", remote + suffix}
		},
		parseStat: svnParseStat,
		latest:    "latest",
		readFile: func(rev, file, remote string) []string {
			return []string{"svn", "cat", "--", remote + "/" + file + "@" + rev}
		},
		// TODO: zip
	},

	"bzr": {
		vcs: "bzr",
		init: func(remote string) []string {
			return []string{"bzr", "branch", "--use-existing-dir", "--", remote, "."}
		},
		fetch: []string{
			"bzr", "pull", "--overwrite-tags",
		},
		tags: func(remote string) []string {
			return []string{"bzr", "tags"}
		},
		tagRE:         re(`(?m)^\S+`),
		badLocalRevRE: re(`^revno:-`),
		statLocal: func(rev, remote string) []string {
			return []string{"bzr", "log", "-l1", "--long", "--show-ids", "-r", rev}
		},
		parseStat: bzrParseStat,
		latest:    "revno:-1",
		readFile: func(rev, file, remote string) []string {
			return []string{"bzr", "cat", "-r", rev, file}
		},
		readZip: func(rev, subdir, remote, target string) []string {
			extra := []string{}
			if subdir != "" {
				extra = []string{"./" + subdir}
			}
			return str.StringList("bzr", "export", "--format=zip", "-r", rev, "--root=prefix/", "--", target, extra)
		},
	},

	"fossil": {
		vcs: "fossil",
		init: func(remote string) []string {
			return []string{"fossil", "clone", "--", remote, ".fossil"}
		},
		fetch: []string{"fossil", "pull", "-R", ".fossil"},
		tags: func(remote string) []string {
			return []string{"fossil", "tag", "-R", ".fossil", "list"}
		},
		tagRE: re(`XXXTODO`),
		statLocal: func(rev, remote string) []string {
			return []string{"fossil", "info", "-R", ".fossil", rev}
		},
		parseStat: fossilParseStat,
		latest:    "trunk",
		readFile: func(rev, file, remote string) []string {
			return []string{"fossil", "cat", "-R", ".fossil", "-r", rev, file}
		},
		readZip: func(rev, subdir, remote, target string) []string {
			extra := []string{}
			if subdir != "" && !strings.ContainsAny(subdir, "*?[],") {
				extra = []string{"--include", subdir}
			}
			// Note that vcsRepo.ReadZip below rewrites this command
			// to run in a different directory, to work around a fossil bug.
			return str.StringList("fossil", "zip", "-R", ".fossil", "--name", "prefix", extra, "--", rev, target)
		},
	},
}

func (r *vcsRepo) loadTags() {
	out, err := Run(r.dir, r.cmd.tags(r.remote))
	if err != nil {
		return
	}

	// Run tag-listing command and extract tags.
	r.tags = make(map[string]bool)
	for _, tag := range r.cmd.tagRE.FindAllString(string(out), -1) {
		if r.cmd.badLocalRevRE != nil && r.cmd.badLocalRevRE.MatchString(tag) {
			continue
		}
		r.tags[tag] = true
	}
}

func (r *vcsRepo) loadBranches() {
	if r.cmd.branches == nil {
		return
	}

	out, err := Run(r.dir, r.cmd.branches(r.remote))
	if err != nil {
		return
	}

	r.branches = make(map[string]bool)
	for _, branch := range r.cmd.branchRE.FindAllString(string(out), -1) {
		if r.cmd.badLocalRevRE != nil && r.cmd.badLocalRevRE.MatchString(branch) {
			continue
		}
		r.branches[branch] = true
	}
}

func (r *vcsRepo) Tags(prefix string) ([]string, error) {
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	r.tagsOnce.Do(r.loadTags)

	tags := []string{}
	for tag := range r.tags {
		if strings.HasPrefix(tag, prefix) {
			tags = append(tags, tag)
		}
	}
	sort.Strings(tags)
	return tags, nil
}

func (r *vcsRepo) Stat(rev string) (*RevInfo, error) {
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	if rev == "latest" {
		rev = r.cmd.latest
	}
	r.branchesOnce.Do(r.loadBranches)
	revOK := (r.cmd.badLocalRevRE == nil || !r.cmd.badLocalRevRE.MatchString(rev)) && !r.branches[rev]
	if revOK {
		if info, err := r.statLocal(rev); err == nil {
			return info, nil
		}
	}

	r.fetchOnce.Do(r.fetch)
	if r.fetchErr != nil {
		return nil, r.fetchErr
	}
	info, err := r.statLocal(rev)
	if err != nil {
		return nil, err
	}
	if !revOK {
		info.Version = info.Name
	}
	return info, nil
}

func (r *vcsRepo) fetch() {
	if len(r.cmd.fetch) > 0 {
		_, r.fetchErr = Run(r.dir, r.cmd.fetch)
	}
}

func (r *vcsRepo) statLocal(rev string) (*RevInfo, error) {
	out, err := Run(r.dir, r.cmd.statLocal(rev, r.remote))
	if err != nil {
		return nil, &UnknownRevisionError{Rev: rev}
	}
	return r.cmd.parseStat(rev, string(out))
}

func (r *vcsRepo) Latest() (*RevInfo, error) {
	return r.Stat("latest")
}

func (r *vcsRepo) ReadFile(rev, file string, maxSize int64) ([]byte, error) {
	if rev == "latest" {
		rev = r.cmd.latest
	}
	_, err := r.Stat(rev) // download rev into local repo
	if err != nil {
		return nil, err
	}

	// r.Stat acquires r.mu, so lock after that.
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	out, err := Run(r.dir, r.cmd.readFile(rev, file, r.remote))
	if err != nil {
		return nil, os.ErrNotExist
	}
	return out, nil
}

func (r *vcsRepo) ReadFileRevs(revs []string, file string, maxSize int64) (map[string]*FileRev, error) {
	// We don't technically need to lock here since we're returning an error
	// uncondititonally, but doing so anyway will help to avoid baking in
	// lock-inversion bugs.
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	return nil, vcsErrorf("ReadFileRevs not implemented")
}

func (r *vcsRepo) RecentTag(rev, prefix, major string) (tag string, err error) {
	// We don't technically need to lock here since we're returning an error
	// uncondititonally, but doing so anyway will help to avoid baking in
	// lock-inversion bugs.
	unlock, err := r.mu.Lock()
	if err != nil {
		return "", err
	}
	defer unlock()

	return "", vcsErrorf("RecentTag not implemented")
}

func (r *vcsRepo) DescendsFrom(rev, tag string) (bool, error) {
	unlock, err := r.mu.Lock()
	if err != nil {
		return false, err
	}
	defer unlock()

	return false, vcsErrorf("DescendsFrom not implemented")
}

func (r *vcsRepo) ReadZip(rev, subdir string, maxSize int64) (zip io.ReadCloser, actualSubdir string, err error) {
	if r.cmd.readZip == nil {
		return nil, "", vcsErrorf("ReadZip not implemented for %s", r.cmd.vcs)
	}

	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, "", err
	}
	defer unlock()

	if rev == "latest" {
		rev = r.cmd.latest
	}
	f, err := ioutil.TempFile("", "go-readzip-*.zip")
	if err != nil {
		return nil, "", err
	}
	if r.cmd.vcs == "fossil" {
		// If you run
		//	fossil zip -R .fossil --name prefix trunk /tmp/x.zip
		// fossil fails with "unable to create directory /tmp" [sic].
		// Change the command to run in /tmp instead,
		// replacing the -R argument with an absolute path.
		args := r.cmd.readZip(rev, subdir, r.remote, filepath.Base(f.Name()))
		for i := range args {
			if args[i] == ".fossil" {
				args[i] = filepath.Join(r.dir, ".fossil")
			}
		}
		_, err = Run(filepath.Dir(f.Name()), args)
	} else {
		_, err = Run(r.dir, r.cmd.readZip(rev, subdir, r.remote, f.Name()))
	}
	if err != nil {
		f.Close()
		os.Remove(f.Name())
		return nil, "", err
	}
	return &deleteCloser{f}, "", nil
}

// deleteCloser is a file that gets deleted on Close.
type deleteCloser struct {
	*os.File
}

func (d *deleteCloser) Close() error {
	defer os.Remove(d.File.Name())
	return d.File.Close()
}

func hgParseStat(rev, out string) (*RevInfo, error) {
	f := strings.Fields(string(out))
	if len(f) < 3 {
		return nil, vcsErrorf("unexpected response from hg log: %q", out)
	}
	hash := f[0]
	version := rev
	if strings.HasPrefix(hash, version) {
		version = hash // extend to full hash
	}
	t, err := strconv.ParseInt(f[1], 10, 64)
	if err != nil {
		return nil, vcsErrorf("invalid time from hg log: %q", out)
	}

	var tags []string
	for _, tag := range f[3:] {
		if tag != "tip" {
			tags = append(tags, tag)
		}
	}
	sort.Strings(tags)

	info := &RevInfo{
		Name:    hash,
		Short:   ShortenSHA1(hash),
		Time:    time.Unix(t, 0).UTC(),
		Version: version,
		Tags:    tags,
	}
	return info, nil
}

func svnParseStat(rev, out string) (*RevInfo, error) {
	var log struct {
		Logentry struct {
			Revision int64  `xml:"revision,attr"`
			Date     string `xml:"date"`
		} `xml:"logentry"`
	}
	if err := xml.Unmarshal([]byte(out), &log); err != nil {
		return nil, vcsErrorf("unexpected response from svn log --xml: %v\n%s", err, out)
	}

	t, err := time.Parse(time.RFC3339, log.Logentry.Date)
	if err != nil {
		return nil, vcsErrorf("unexpected response from svn log --xml: %v\n%s", err, out)
	}

	info := &RevInfo{
		Name:    fmt.Sprintf("%d", log.Logentry.Revision),
		Short:   fmt.Sprintf("%012d", log.Logentry.Revision),
		Time:    t.UTC(),
		Version: rev,
	}
	return info, nil
}

func bzrParseStat(rev, out string) (*RevInfo, error) {
	var revno int64
	var tm time.Time
	for _, line := range strings.Split(out, "\n") {
		if line == "" || line[0] == ' ' || line[0] == '\t' {
			// End of header, start of commit message.
			break
		}
		if line[0] == '-' {
			continue
		}
		i := strings.Index(line, ":")
		if i < 0 {
			// End of header, start of commit message.
			break
		}
		key, val := line[:i], strings.TrimSpace(line[i+1:])
		switch key {
		case "revno":
			if j := strings.Index(val, " "); j >= 0 {
				val = val[:j]
			}
			i, err := strconv.ParseInt(val, 10, 64)
			if err != nil {
				return nil, vcsErrorf("unexpected revno from bzr log: %q", line)
			}
			revno = i
		case "timestamp":
			j := strings.Index(val, " ")
			if j < 0 {
				return nil, vcsErrorf("unexpected timestamp from bzr log: %q", line)
			}
			t, err := time.Parse("2006-01-02 15:04:05 -0700", val[j+1:])
			if err != nil {
				return nil, vcsErrorf("unexpected timestamp from bzr log: %q", line)
			}
			tm = t.UTC()
		}
	}
	if revno == 0 || tm.IsZero() {
		return nil, vcsErrorf("unexpected response from bzr log: %q", out)
	}

	info := &RevInfo{
		Name:    fmt.Sprintf("%d", revno),
		Short:   fmt.Sprintf("%012d", revno),
		Time:    tm,
		Version: rev,
	}
	return info, nil
}

func fossilParseStat(rev, out string) (*RevInfo, error) {
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "uuid:") {
			f := strings.Fields(line)
			if len(f) != 5 || len(f[1]) != 40 || f[4] != "UTC" {
				return nil, vcsErrorf("unexpected response from fossil info: %q", line)
			}
			t, err := time.Parse("2006-01-02 15:04:05", f[2]+" "+f[3])
			if err != nil {
				return nil, vcsErrorf("unexpected response from fossil info: %q", line)
			}
			hash := f[1]
			version := rev
			if strings.HasPrefix(hash, version) {
				version = hash // extend to full hash
			}
			info := &RevInfo{
				Name:    hash,
				Short:   ShortenSHA1(hash),
				Time:    t,
				Version: version,
			}
			return info, nil
		}
	}
	return nil, vcsErrorf("unexpected response from fossil info: %q", out)
}
