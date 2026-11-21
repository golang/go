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
	"unicode"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/vcscmd"
	"cmd/internal/par"

	"golang.org/x/mod/semver"
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

	statCache     par.ErrCache[string, *RevInfo]  // cache key is revision
	readFileCache par.ErrCache[[2]string, []byte] // cache key is revision and file path
}

func newVCSRepo(ctx context.Context, vcs, remote string, local bool) (Repo, error) {
	if !local && (strings.HasPrefix(remote, "-") || strings.ContainsFunc(remote, unicode.IsControl)) {
		// A remote may be written to .hg/hgrc. Local directory names
		// are not written there and may contain control characters.
		return nil, fmt.Errorf("invalid vcs remote: %s %q", vcs, remote)
	}
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
		init := cmd.init(r.remote)
		_, err = run(ctx, r.dir, init)
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
	vcs                string                                                // vcs name "hg"
	init               func(remote string) *vcscmd.Command                   // cmd to init repo to track remote
	postInit           func(context.Context, *vcsRepo) error                 // func to init repo after .init runs
	repoSum            func(remote string) *vcscmd.Command                   // cmd to calculate reposum of remote repo
	lookupRef          func(remote, ref string) *vcscmd.Command              // cmd to look up ref in remote repo
	tags               func(remote string) *vcscmd.Command                   // cmd to list local tags
	tagsNeedsFetch     bool                                                  // run fetch before tags
	tagRE              *lazyregexp.Regexp                                    // regexp to extract tag names from output of tags cmd
	branches           func(remote string) *vcscmd.Command                   // cmd to list local branches
	branchesNeedsFetch bool                                                  // run branches before tags
	branchRE           *lazyregexp.Regexp                                    // regexp to extract branch names from output of tags cmd
	badLocalRevRE      *lazyregexp.Regexp                                    // regexp of names that must not be served out of local cache without doing fetch first
	statLocal          func(rev, remote string) *vcscmd.Command              // cmd to stat local rev
	parseStat          func(rev, out string) (*RevInfo, error)               // func to parse output of statLocal
	fetch              func() *vcscmd.Command                                // cmd to fetch everything from remote
	latest             string                                                // name of latest commit on remote (tip, HEAD, etc)
	descendsFrom       func(rev, tag string) *vcscmd.Command                 // cmd to check whether rev descends from tag
	recentTags         func(rev string) *vcscmd.Command                      // cmd to print tag ancestors of rev
	readFile           func(rev, file, remote string) *vcscmd.Command        // cmd to read rev's file
	readZip            func(rev, subdir, dir, target string) *vcscmd.Command // cmd to read rev's subdir of repo in dir as zip file

	// arbitrary function to read rev's subdir as zip file
	doReadZip func(ctx context.Context, dst io.Writer, workDir, rev, subdir, remote string) error
}

var re = lazyregexp.New

var vcsCmds = map[string]*vcsCmd{
	"hg": {
		vcs: "hg",
		repoSum: func(remote string) *vcscmd.Command {
			return vcscmd.New("hg").Option("--config", hgRepoSumExt()).Add("goreposum").Operands(remote)
		},
		lookupRef: func(remote, ref string) *vcscmd.Command {
			return vcscmd.New("hg").Option("--config", hgRepoSumExt()).Add("golookup").Operands(remote, ref)
		},
		init: func(remote string) *vcscmd.Command {
			return vcscmd.New("hg", "init", ".")
		},
		postInit: hgAddRemote,
		tags: func(remote string) *vcscmd.Command {
			return vcscmd.New("hg", "tags", "-q")
		},
		tagsNeedsFetch: true,
		tagRE:          re(`(?m)^[^\n]+$`),
		branches: func(remote string) *vcscmd.Command {
			return vcscmd.New("hg", "branches", "-c", "-q")
		},
		branchesNeedsFetch: true,
		branchRE:           re(`(?m)^[^\n]+$`),
		badLocalRevRE:      re(`(?m)^(tip)$`),
		statLocal: func(rev, remote string) *vcscmd.Command {
			return hgLog(hgQuote(rev)).Add("-l1", "--template", "{node} {date|hgdate} {tags}")
		},
		parseStat: hgParseStat,
		fetch: func() *vcscmd.Command {
			return vcscmd.New("hg", "pull", "-f")
		},
		latest: "tip",
		descendsFrom: func(rev, tag string) *vcscmd.Command {
			return hgLog("ancestors(" + hgQuote(rev) + ") and " + hgQuote(tag))
		},
		recentTags: func(rev string) *vcscmd.Command {
			return hgLog("ancestors("+hgQuote(rev)+") and tag()").Add("--template", "{tags}\n")
		},
		readFile: func(rev, file, remote string) *vcscmd.Command {
			return vcscmd.New("hg", "cat").Option("--rev", rev).Operands(file)
		},
		readZip: func(rev, subdir, dir, target string) *vcscmd.Command {
			c := vcscmd.New("hg", "archive", "-t", "zip", "--no-decode").Option("--rev", rev).Add("--prefix=prefix/")
			if subdir != "" {
				c.Option("--include", "path:"+subdir)
			}
			return c.Operands(target)
		},
	},

	"svn": {
		vcs:  "svn",
		init: nil, // no local checkout
		tags: func(remote string) *vcscmd.Command {
			return vcscmd.New("svn", "list").Operands(strings.TrimSuffix(remote, "/trunk") + "/tags")
		},
		tagRE: re(`(?m)^(.*?)/?$`),
		statLocal: func(rev, remote string) *vcscmd.Command {
			suffix := "@" + rev
			if rev == "latest" {
				suffix = "@"
			}
			return vcscmd.New("svn", "log", "-l1", "--xml").Operands(remote + suffix)
		},
		parseStat: svnParseStat,
		latest:    "latest",
		readFile: func(rev, file, remote string) *vcscmd.Command {
			return vcscmd.New("svn", "cat").Operands(remote + "/" + file + "@" + rev)
		},
		doReadZip: svnReadZip,
	},

	"fossil": {
		vcs: "fossil",
		init: func(remote string) *vcscmd.Command {
			return vcscmd.New("fossil", "clone").Operands(remote, ".fossil")
		},
		fetch: func() *vcscmd.Command {
			return vcscmd.New("fossil", "pull", "-R", ".fossil")
		},
		tags: func(remote string) *vcscmd.Command {
			return vcscmd.New("fossil", "tag", "-R", ".fossil", "list")
		},
		tagRE: re(`XXXTODO`),
		statLocal: func(rev, remote string) *vcscmd.Command {
			return vcscmd.New("fossil", "info", "-R", ".fossil").Operands(rev)
		},
		parseStat: fossilParseStat,
		latest:    "trunk",
		readFile: func(rev, file, remote string) *vcscmd.Command {
			return vcscmd.New("fossil", "cat", "-R", ".fossil").Option("-r", rev).Operands(file)
		},
		readZip: func(rev, subdir, dir, target string) *vcscmd.Command {
			// ReadZip runs fossil in the output directory, not r.dir.
			c := vcscmd.New("fossil", "zip").Option("-R", filepath.Join(dir, ".fossil")).Add("--name", "prefix")
			if subdir != "" && !strings.ContainsAny(subdir, "*?[],") {
				c.Option("--include", subdir)
			}
			return c.Operands(rev, target)
		},
	},
}

func hgRepoSumExt() string {
	return "extensions.goreposum=" + filepath.Join(cfg.GOROOT, "lib/hg/goreposum.py")
}

func hgLog(revset string) *vcscmd.Command {
	// Hg first tries the whole revset as a revision name. A leading space
	// (not valid at the start of a name) forces it to parse the expression.
	return vcscmd.New("hg", "log").Option("--rev", " "+revset)
}

var hgRevsetEscaper = strings.NewReplacer(`\`, `\\`, `"`, `\"`)

// hgQuote quotes a single revision symbol for use in a revset.
func hgQuote(rev string) string {
	return `"` + hgRevsetEscaper.Replace(rev) + `"`
}

func (r *vcsRepo) loadTags(ctx context.Context) {
	if r.cmd.tagsNeedsFetch {
		r.fetchOnce.Do(func() { r.fetch(ctx) })
	}

	cmd := r.cmd.tags(r.remote)
	out, err := run(ctx, r.dir, cmd)
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

	cmd := r.cmd.branches(r.remote)
	out, err := run(ctx, r.dir, cmd)
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
	cmd := r.cmd.repoSum(where)
	out, err := run(ctx, r.dir, cmd)
	if err != nil {
		return
	}
	r.repoSum = strings.TrimSpace(string(out))
}

func (r *vcsRepo) lookupRef(ctx context.Context, ref string) (string, error) {
	if r.cmd.lookupRef == nil {
		return "", fmt.Errorf("no lookupRef")
	}
	cmd := r.cmd.lookupRef(r.remote, ref)
	out, err := run(ctx, r.dir, cmd)
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
	return r.statCache.Do(rev, func() (*RevInfo, error) {
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
	})
}

func (r *vcsRepo) fetch(ctx context.Context) {
	if r.cmd.fetch != nil {
		release, err := base.AcquireNet()
		if err != nil {
			r.fetchErr = err
			return
		}
		cmd := r.cmd.fetch()
		_, r.fetchErr = run(ctx, r.dir, cmd)
		release()
		r.fetched.Store(true)
	}
}

func (r *vcsRepo) statLocal(ctx context.Context, rev string) (*RevInfo, error) {
	cmd := r.cmd.statLocal(rev, r.remote)
	out, err := run(ctx, r.dir, cmd)
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
	return r.readFileCache.Do([2]string{rev, file}, func() ([]byte, error) {
		if rev == "latest" {
			rev = r.cmd.latest
		}
		info, err := r.Stat(ctx, rev) // download rev into local repo
		if err != nil {
			return nil, err
		}
		if r.cmd.vcs == "hg" {
			rev = info.Name
		}

		// r.Stat acquires r.mu, so lock after that.
		unlock, err := r.mu.Lock()
		if err != nil {
			return nil, err
		}
		defer unlock()

		cmd := r.cmd.readFile(rev, file, r.remote)
		out, err := run(ctx, r.dir, cmd)
		if err != nil {
			return nil, fs.ErrNotExist
		}
		return out, nil
	})
}

func (r *vcsRepo) RecentTag(ctx context.Context, rev, prefix string, allowed func(string) bool) (tag string, err error) {
	// Only lock for the subprocess execution, not for the tag scan.
	// allowed may call other methods that acquire the lock.
	unlock, err := r.mu.Lock()
	if err != nil {
		return "", err
	}

	if r.cmd.recentTags == nil {
		unlock()
		return "", vcsErrorf("vcs %s: RecentTag: %w", r.cmd.vcs, errors.ErrUnsupported)
	}
	cmd := r.cmd.recentTags(rev)
	out, err := run(ctx, r.dir, cmd)
	unlock()
	if err != nil {
		return "", err
	}

	highest := ""
	for _, tag := range strings.Fields(string(out)) {
		if !strings.HasPrefix(tag, prefix) || !allowed(tag) {
			continue
		}
		semtag := tag[len(prefix):]
		if semver.Compare(semtag, highest) > 0 {
			highest = semtag
		}
	}
	if highest != "" {
		return prefix + highest, nil
	}
	return "", nil
}

func (r *vcsRepo) DescendsFrom(ctx context.Context, rev, tag string) (bool, error) {
	unlock, err := r.mu.Lock()
	if err != nil {
		return false, err
	}
	defer unlock()

	if r.cmd.descendsFrom == nil {
		return false, vcsErrorf("vcs %s: DescendsFrom: %w", r.cmd.vcs, errors.ErrUnsupported)
	}

	cmd := r.cmd.descendsFrom(rev, tag)
	out, err := run(ctx, r.dir, cmd)
	if err != nil {
		return false, err
	}
	return strings.TrimSpace(string(out)) != "", nil
}

func (r *vcsRepo) ReadZip(ctx context.Context, rev, subdir string, maxSize int64) (zip io.ReadCloser, err error) {
	if r.cmd.readZip == nil && r.cmd.doReadZip == nil {
		return nil, vcsErrorf("vcs %s: ReadZip: %w", r.cmd.vcs, errors.ErrUnsupported)
	}

	if rev == "latest" {
		rev = r.cmd.latest
	}
	info, err := r.Stat(ctx, rev) // download rev into local repo
	if err != nil {
		return nil, err
	}
	if r.cmd.vcs == "hg" {
		rev = info.Name
	}

	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

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
		// Run the command in /tmp instead.
		cmd := r.cmd.readZip(rev, subdir, r.dir, filepath.Base(f.Name()))
		_, err = run(ctx, filepath.Dir(f.Name()), cmd)
	} else {
		cmd := r.cmd.readZip(rev, subdir, r.dir, f.Name())
		_, err = run(ctx, r.dir, cmd)
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
