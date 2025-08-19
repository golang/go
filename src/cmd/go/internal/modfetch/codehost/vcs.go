// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"context"
	"errors"
	"fmt"
	"internal/lazyregexp"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/str"
	"cmd/internal/par"
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

func (e *VCSError) Unwrap() error { return e.Err }

func vcsErrorf(format string, a ...any) error {
	return &VCSError{Err: fmt.Errorf(format, a...)}
}

type vcsCacheKey struct {
	vcs    string
	remote string
	local  bool
}

func NewRepo(ctx context.Context, vcs, remote string, local bool) (Repo, error) {
	return vcsRepoCache.Do(vcsCacheKey{vcs, remote, local}, func() (Repo, error) {
		repo, err := newVCSRepo(ctx, vcs, remote, local)
		if err != nil {
			return nil, &VCSError{err}
		}
		return repo, nil
	})
}

var vcsRepoCache par.ErrCache[vcsCacheKey, Repo]

type vcsRepo struct {
	mu lockedfile.Mutex // protects all commands, so we don't have to decide which are safe on a per-VCS basis

	remote string
	cmd    *vcsCmd
	dir    string
	local  bool

	tagsOnce sync.Once
	tags     map[string]bool

	branchesOnce sync.Once
	branches     map[string]bool

	fetchOnce sync.Once
	fetchErr  error
	fetched   atomic.Bool

	repoSumOnce sync.Once
	repoSum     string
}

func newVCSRepo(ctx context.Context, vcs, remote string, local bool) (Repo, error) {
	if vcs == "git" {
		return newGitRepo(ctx, remote, local)
	}
	r := &vcsRepo{remote: remote, local: local}
	cmd := vcsCmds[vcs]
	if cmd == nil {
		return nil, fmt.Errorf("unknown vcs: %s %s", vcs, remote)
	}
	r.cmd = cmd
	if local {
		info, err := os.Stat(remote)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			return nil, fmt.Errorf("%s exists but is not a directory", remote)
		}
		r.dir = remote
		r.mu.Path = r.dir + ".lock"
		return r, nil
	}
	if !strings.Contains(remote, "://") {
		return nil, fmt.Errorf("invalid vcs remote: %s %s", vcs, remote)
	}
	var err error
	r.dir, r.mu.Path, err = WorkDir(ctx, vcsWorkDirType+vcs, r.remote)
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
		release, err := base.AcquireNet()
		if err != nil {
			return nil, err
		}
		_, err = Run(ctx, r.dir, cmd.init(r.remote))
		if err == nil && cmd.postInit != nil {
			err = cmd.postInit(ctx, r)
		}
		release()

		if err != nil {
			os.RemoveAll(r.dir)
			return nil, err
		}
	}
	return r, nil
}

const vcsWorkDirType = "vcs1."

type vcsCmd struct {
	vcs                string                                            // vcs name "hg"
	init               func(remote string) []string                      // cmd to init repo to track remote
	postInit           func(context.Context, *vcsRepo) error             // func to init repo after .init runs
	repoSum            func(remote string) []string                      // cmd to calculate reposum of remote repo
	lookupRef          func(remote, ref string) []string                 // cmd to look up ref in remote repo
	tags               func(remote string) []string                      // cmd to list local tags
	tagsNeedsFetch     bool                                              // run fetch before tags
	tagRE              *lazyregexp.Regexp                                // regexp to extract tag names from output of tags cmd
	branches           func(remote string) []string                      // cmd to list local branches
	branchesNeedsFetch bool                                              // run branches before tags
	branchRE           *lazyregexp.Regexp                                // regexp to extract branch names from output of tags cmd
	badLocalRevRE      *lazyregexp.Regexp                                // regexp of names that must not be served out of local cache without doing fetch first
	statLocal          func(rev, remote string) []string                 // cmd to stat local rev
	parseStat          func(rev, out string) (*RevInfo, error)           // func to parse output of statLocal
	fetch              []string                                          // cmd to fetch everything from remote
	latest             string                                            // name of latest commit on remote (tip, HEAD, etc)
	readFile           func(rev, file, remote string) []string           // cmd to read rev's file
	readZip            func(rev, subdir, remote, target string) []string // cmd to read rev's subdir as zip file

	// arbitrary function to read rev's subdir as zip file
	doReadZip func(ctx context.Context, dst io.Writer, workDir, rev, subdir, remote string) error
}

var re = lazyregexp.New

var vcsCmds = map[string]*vcsCmd{
	"hg": {
		vcs: "hg",
		repoSum: func(remote string) []string {
			return []string{
				"hg",
				"--config=extensions.goreposum=" + filepath.Join(cfg.GOROOT, "lib/hg/goreposum.py"),
				"goreposum",
				remote,
			}
		},
		lookupRef: func(remote, ref string) []string {
			return []string{
				"hg",
				"--config=extensions.goreposum=" + filepath.Join(cfg.GOROOT, "lib/hg/goreposum.py"),
				"golookup",
				remote,
				ref,
			}
		},
		init: func(remote string) []string {
			return []string{"hg", "init", "."}
		},
		postInit: hgAddRemote,
		tags: func(remote string) []string {
			return []string{"hg", "tags", "-q"}
		},
		tagsNeedsFetch: true,
		tagRE:          re(`(?m)^[^\n]+$`),
		branches: func(remote string) []string {
			return []string{"hg", "branches", "-c", "-q"}
		},
		branchesNeedsFetch: true,
		branchRE:           re(`(?m)^[^\n]+$`),
		badLocalRevRE:      re(`(?m)^(tip)$`),
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
		doReadZip: svnReadZip,
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

func (r *vcsRepo) loadTags(ctx context.Context) {
	if r.cmd.tagsNeedsFetch {
		r.fetchOnce.Do(func() { r.fetch(ctx) })
	}

	out, err := Run(ctx, r.dir, r.cmd.tags(r.remote))
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

func (r *vcsRepo) loadBranches(ctx context.Context) {
	if r.cmd.branches == nil {
		return
	}

	if r.cmd.branchesNeedsFetch {
		r.fetchOnce.Do(func() { r.fetch(ctx) })
	}

	out, err := Run(ctx, r.dir, r.cmd.branches(r.remote))
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

func (r *vcsRepo) loadRepoSum(ctx context.Context) {
	if r.cmd.repoSum == nil {
		return
	}
	where := r.remote
	if r.fetched.Load() {
		where = "." // use local repo
	}
	out, err := Run(ctx, r.dir, r.cmd.repoSum(where))
	if err != nil {
		return
	}
	r.repoSum = strings.TrimSpace(string(out))
}

func (r *vcsRepo) lookupRef(ctx context.Context, ref string) (string, error) {
	if r.cmd.lookupRef == nil {
		return "", fmt.Errorf("no lookupRef")
	}
	out, err := Run(ctx, r.dir, r.cmd.lookupRef(r.remote, ref))
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// repoSumOrigin returns an Origin containing a RepoSum.
func (r *vcsRepo) repoSumOrigin(ctx context.Context) *Origin {
	origin := &Origin{
		VCS:     r.cmd.vcs,
		URL:     r.remote,
		RepoSum: r.repoSum,
	}
	r.repoSumOnce.Do(func() { r.loadRepoSum(ctx) })
	origin.RepoSum = r.repoSum
	return origin
}

func (r *vcsRepo) CheckReuse(ctx context.Context, old *Origin, subdir string) error {
	if old == nil {
		return fmt.Errorf("missing origin")
	}
	if old.VCS != r.cmd.vcs || old.URL != r.remote {
		return fmt.Errorf("origin moved from %v %q to %v %q", old.VCS, old.URL, r.cmd.vcs, r.remote)
	}
	if old.Subdir != subdir {
		return fmt.Errorf("origin moved from %v %q %q to %v %q %q", old.VCS, old.URL, old.Subdir, r.cmd.vcs, r.remote, subdir)
	}

	if old.Ref == "" && old.RepoSum == "" && old.Hash != "" {
		// Hash has to remain in repo.
		hash, err := r.lookupRef(ctx, old.Hash)
		if err == nil && hash == old.Hash {
			return nil
		}
		if err != nil {
			return fmt.Errorf("looking up hash: %v", err)
		}
		return fmt.Errorf("hash changed") // weird but maybe they made a tag
	}

	if old.Ref != "" && old.RepoSum == "" {
		hash, err := r.lookupRef(ctx, old.Ref)
		if err == nil && hash != "" && hash == old.Hash {
			return nil
		}
	}

	r.repoSumOnce.Do(func() { r.loadRepoSum(ctx) })
	if r.repoSum != "" {
		if old.RepoSum == "" {
			return fmt.Errorf("non-specific origin")
		}
		if old.RepoSum != r.repoSum {
			return fmt.Errorf("repo changed")
		}
		return nil
	}
	return fmt.Errorf("vcs %s: CheckReuse: %w", r.cmd.vcs, errors.ErrUnsupported)
}

func (r *vcsRepo) Tags(ctx context.Context, prefix string) (*Tags, error) {
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	r.tagsOnce.Do(func() { r.loadTags(ctx) })
	tags := &Tags{
		Origin: r.repoSumOrigin(ctx),
		List:   []Tag{},
	}
	for tag := range r.tags {
		if strings.HasPrefix(tag, prefix) {
			tags.List = append(tags.List, Tag{tag, ""})
		}
	}
	sort.Slice(tags.List, func(i, j int) bool {
		return tags.List[i].Name < tags.List[j].Name
	})
	return tags, nil
}

func (r *vcsRepo) Stat(ctx context.Context, rev string) (*RevInfo, error) {
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	if rev == "latest" {
		rev = r.cmd.latest
	}
	r.branchesOnce.Do(func() { r.loadBranches(ctx) })
	if r.local {
		// Ignore the badLocalRevRE precondition in local only mode.
		// We cannot fetch latest upstream changes so only serve what's in the local cache.
		return r.statLocal(ctx, rev)
	}
	revOK := (r.cmd.badLocalRevRE == nil || !r.cmd.badLocalRevRE.MatchString(rev)) && !r.branches[rev]
	if revOK {
		if info, err := r.statLocal(ctx, rev); err == nil {
			return info, nil
		}
	}

	r.fetchOnce.Do(func() { r.fetch(ctx) })
	if r.fetchErr != nil {
		return nil, r.fetchErr
	}
	info, err := r.statLocal(ctx, rev)
	if err != nil {
		return info, err
	}
	if !revOK {
		info.Version = info.Name
	}
	return info, nil
}

func (r *vcsRepo) fetch(ctx context.Context) {
	if len(r.cmd.fetch) > 0 {
		release, err := base.AcquireNet()
		if err != nil {
			r.fetchErr = err
			return
		}
		_, r.fetchErr = Run(ctx, r.dir, r.cmd.fetch)
		release()
		r.fetched.Store(true)
	}
}

func (r *vcsRepo) statLocal(ctx context.Context, rev string) (*RevInfo, error) {
	out, err := Run(ctx, r.dir, r.cmd.statLocal(rev, r.remote))
	if err != nil {
		info := &RevInfo{Origin: r.repoSumOrigin(ctx)}
		return info, &UnknownRevisionError{Rev: rev}
	}
	info, err := r.cmd.parseStat(rev, string(out))
	if err != nil {
		return nil, err
	}
	if info.Origin == nil {
		info.Origin = new(Origin)
	}
	info.Origin.VCS = r.cmd.vcs
	info.Origin.URL = r.remote
	info.Origin.Ref = rev
	if strings.HasPrefix(info.Name, rev) && len(rev) >= 12 {
		info.Origin.Ref = "" // duplicates Hash
	}
	return info, nil
}

func (r *vcsRepo) Latest(ctx context.Context) (*RevInfo, error) {
	return r.Stat(ctx, "latest")
}

func (r *vcsRepo) ReadFile(ctx context.Context, rev, file string, maxSize int64) ([]byte, error) {
	if rev == "latest" {
		rev = r.cmd.latest
	}
	_, err := r.Stat(ctx, rev) // download rev into local repo
	if err != nil {
		return nil, err
	}

	// r.Stat acquires r.mu, so lock after that.
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	out, err := Run(ctx, r.dir, r.cmd.readFile(rev, file, r.remote))
	if err != nil {
		return nil, fs.ErrNotExist
	}
	return out, nil
}

func (r *vcsRepo) RecentTag(ctx context.Context, rev, prefix string, allowed func(string) bool) (tag string, err error) {
	// We don't technically need to lock here since we're returning an error
	// unconditionally, but doing so anyway will help to avoid baking in
	// lock-inversion bugs.
	unlock, err := r.mu.Lock()
	if err != nil {
		return "", err
	}
	defer unlock()

	return "", vcsErrorf("vcs %s: RecentTag: %w", r.cmd.vcs, errors.ErrUnsupported)
}

func (r *vcsRepo) DescendsFrom(ctx context.Context, rev, tag string) (bool, error) {
	unlock, err := r.mu.Lock()
	if err != nil {
		return false, err
	}
	defer unlock()

	return false, vcsErrorf("vcs %s: DescendsFrom: %w", r.cmd.vcs, errors.ErrUnsupported)
}

func (r *vcsRepo) ReadZip(ctx context.Context, rev, subdir string, maxSize int64) (zip io.ReadCloser, err error) {
	if r.cmd.readZip == nil && r.cmd.doReadZip == nil {
		return nil, vcsErrorf("vcs %s: ReadZip: %w", r.cmd.vcs, errors.ErrUnsupported)
	}

	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	if rev == "latest" {
		rev = r.cmd.latest
	}
	f, err := os.CreateTemp("", "go-readzip-*.zip")
	if err != nil {
		return nil, err
	}
	if r.cmd.doReadZip != nil {
		lw := &limitedWriter{
			W:               f,
			N:               maxSize,
			ErrLimitReached: errors.New("ReadZip: encoded file exceeds allowed size"),
		}
		err = r.cmd.doReadZip(ctx, lw, r.dir, rev, subdir, r.remote)
		if err == nil {
			_, err = f.Seek(0, io.SeekStart)
		}
	} else if r.cmd.vcs == "fossil" {
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
		_, err = Run(ctx, filepath.Dir(f.Name()), args)
	} else {
		_, err = Run(ctx, r.dir, r.cmd.readZip(rev, subdir, r.remote, f.Name()))
	}
	if err != nil {
		f.Close()
		os.Remove(f.Name())
		return nil, err
	}
	return &deleteCloser{f}, nil
}

// deleteCloser is a file that gets deleted on Close.
type deleteCloser struct {
	*os.File
}

func (d *deleteCloser) Close() error {
	defer os.Remove(d.File.Name())
	return d.File.Close()
}

func hgAddRemote(ctx context.Context, r *vcsRepo) error {
	// Write .hg/hgrc with remote URL in it.
	return os.WriteFile(filepath.Join(r.dir, ".hg/hgrc"), []byte(fmt.Sprintf("[paths]\ndefault = %s\n", r.remote)), 0666)
}

func hgParseStat(rev, out string) (*RevInfo, error) {
	f := strings.Fields(out)
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
		Origin:  &Origin{Hash: hash},
		Name:    hash,
		Short:   ShortenSHA1(hash),
		Time:    time.Unix(t, 0).UTC(),
		Version: version,
		Tags:    tags,
	}
	return info, nil
}

func bzrParseStat(rev, out string) (*RevInfo, error) {
	var revno int64
	var tm time.Time
	var tags []string
	for line := range strings.SplitSeq(out, "\n") {
		if line == "" || line[0] == ' ' || line[0] == '\t' {
			// End of header, start of commit message.
			break
		}
		if line[0] == '-' {
			continue
		}
		before, after, found := strings.Cut(line, ":")
		if !found {
			// End of header, start of commit message.
			break
		}
		key, val := before, strings.TrimSpace(after)
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
		case "tags":
			tags = strings.Split(val, ", ")
		}
	}
	if revno == 0 || tm.IsZero() {
		return nil, vcsErrorf("unexpected response from bzr log: %q", out)
	}

	info := &RevInfo{
		Name:    strconv.FormatInt(revno, 10),
		Short:   fmt.Sprintf("%012d", revno),
		Time:    tm,
		Version: rev,
		Tags:    tags,
	}
	return info, nil
}

func fossilParseStat(rev, out string) (*RevInfo, error) {
	for line := range strings.SplitSeq(out, "\n") {
		if strings.HasPrefix(line, "uuid:") || strings.HasPrefix(line, "hash:") {
			f := strings.Fields(line)
			if len(f) != 5 || len(f[1]) != 40 || f[4] != "UTC" {
				return nil, vcsErrorf("unexpected response from fossil info: %q", line)
			}
			t, err := time.Parse(time.DateTime, f[2]+" "+f[3])
			if err != nil {
				return nil, vcsErrorf("unexpected response from fossil info: %q", line)
			}
			hash := f[1]
			version := rev
			if strings.HasPrefix(hash, version) {
				version = hash // extend to full hash
			}
			info := &RevInfo{
				Origin:  &Origin{Hash: hash},
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

type limitedWriter struct {
	W               io.Writer
	N               int64
	ErrLimitReached error
}

func (l *limitedWriter) Write(p []byte) (n int, err error) {
	if l.N > 0 {
		max := len(p)
		if l.N < int64(max) {
			max = int(l.N)
		}
		n, err = l.W.Write(p[:max])
		l.N -= int64(n)
		if err != nil || n >= len(p) {
			return n, err
		}
	}

	return n, l.ErrLimitReached
}
