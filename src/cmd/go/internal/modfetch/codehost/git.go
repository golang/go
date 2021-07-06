// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"bytes"
	"errors"
	"fmt"
	exec "internal/execabs"
	"io"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/lockedfile"
	"cmd/go/internal/par"
	"cmd/go/internal/web"

	"golang.org/x/mod/semver"
)

// LocalGitRepo is like Repo but accepts both Git remote references
// and paths to repositories on the local file system.
func LocalGitRepo(remote string) (Repo, error) {
	return newGitRepoCached(remote, true)
}

// A notExistError wraps another error to retain its original text
// but makes it opaquely equivalent to fs.ErrNotExist.
type notExistError struct {
	err error
}

func (e notExistError) Error() string   { return e.err.Error() }
func (notExistError) Is(err error) bool { return err == fs.ErrNotExist }

const gitWorkDirType = "git3"

var gitRepoCache par.Cache

func newGitRepoCached(remote string, localOK bool) (Repo, error) {
	type key struct {
		remote  string
		localOK bool
	}
	type cached struct {
		repo Repo
		err  error
	}

	c := gitRepoCache.Do(key{remote, localOK}, func() interface{} {
		repo, err := newGitRepo(remote, localOK)
		return cached{repo, err}
	}).(cached)

	return c.repo, c.err
}

func newGitRepo(remote string, localOK bool) (Repo, error) {
	r := &gitRepo{remote: remote}
	if strings.Contains(remote, "://") {
		// This is a remote path.
		var err error
		r.dir, r.mu.Path, err = WorkDir(gitWorkDirType, r.remote)
		if err != nil {
			return nil, err
		}

		unlock, err := r.mu.Lock()
		if err != nil {
			return nil, err
		}
		defer unlock()

		if _, err := os.Stat(filepath.Join(r.dir, "objects")); err != nil {
			if _, err := Run(r.dir, "git", "init", "--bare"); err != nil {
				os.RemoveAll(r.dir)
				return nil, err
			}
			// We could just say git fetch https://whatever later,
			// but this lets us say git fetch origin instead, which
			// is a little nicer. More importantly, using a named remote
			// avoids a problem with Git LFS. See golang.org/issue/25605.
			if _, err := Run(r.dir, "git", "remote", "add", "origin", "--", r.remote); err != nil {
				os.RemoveAll(r.dir)
				return nil, err
			}
		}
		r.remoteURL = r.remote
		r.remote = "origin"
	} else {
		// Local path.
		// Disallow colon (not in ://) because sometimes
		// that's rcp-style host:path syntax and sometimes it's not (c:\work).
		// The go command has always insisted on URL syntax for ssh.
		if strings.Contains(remote, ":") {
			return nil, fmt.Errorf("git remote cannot use host:path syntax")
		}
		if !localOK {
			return nil, fmt.Errorf("git remote must not be local directory")
		}
		r.local = true
		info, err := os.Stat(remote)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			return nil, fmt.Errorf("%s exists but is not a directory", remote)
		}
		r.dir = remote
		r.mu.Path = r.dir + ".lock"
	}
	return r, nil
}

type gitRepo struct {
	remote, remoteURL string
	local             bool
	dir               string

	mu lockedfile.Mutex // protects fetchLevel and git repo state

	fetchLevel int

	statCache par.Cache

	refsOnce sync.Once
	// refs maps branch and tag refs (e.g., "HEAD", "refs/heads/master")
	// to commits (e.g., "37ffd2e798afde829a34e8955b716ab730b2a6d6")
	refs    map[string]string
	refsErr error

	localTagsOnce sync.Once
	localTags     map[string]bool
}

const (
	// How much have we fetched into the git repo (in this process)?
	fetchNone = iota // nothing yet
	fetchSome        // shallow fetches of individual hashes
	fetchAll         // "fetch -t origin": get all remote branches and tags
)

// loadLocalTags loads tag references from the local git cache
// into the map r.localTags.
// Should only be called as r.localTagsOnce.Do(r.loadLocalTags).
func (r *gitRepo) loadLocalTags() {
	// The git protocol sends all known refs and ls-remote filters them on the client side,
	// so we might as well record both heads and tags in one shot.
	// Most of the time we only care about tags but sometimes we care about heads too.
	out, err := Run(r.dir, "git", "tag", "-l")
	if err != nil {
		return
	}

	r.localTags = make(map[string]bool)
	for _, line := range strings.Split(string(out), "\n") {
		if line != "" {
			r.localTags[line] = true
		}
	}
}

// loadRefs loads heads and tags references from the remote into the map r.refs.
// Should only be called as r.refsOnce.Do(r.loadRefs).
func (r *gitRepo) loadRefs() {
	// The git protocol sends all known refs and ls-remote filters them on the client side,
	// so we might as well record both heads and tags in one shot.
	// Most of the time we only care about tags but sometimes we care about heads too.
	out, gitErr := Run(r.dir, "git", "ls-remote", "-q", r.remote)
	if gitErr != nil {
		if rerr, ok := gitErr.(*RunError); ok {
			if bytes.Contains(rerr.Stderr, []byte("fatal: could not read Username")) {
				rerr.HelpText = "Confirm the import path was entered correctly.\nIf this is a private repository, see https://golang.org/doc/faq#git_https for additional information."
			}
		}

		// If the remote URL doesn't exist at all, ideally we should treat the whole
		// repository as nonexistent by wrapping the error in a notExistError.
		// For HTTP and HTTPS, that's easy to detect: we'll try to fetch the URL
		// ourselves and see what code it serves.
		if u, err := url.Parse(r.remoteURL); err == nil && (u.Scheme == "http" || u.Scheme == "https") {
			if _, err := web.GetBytes(u); errors.Is(err, fs.ErrNotExist) {
				gitErr = notExistError{gitErr}
			}
		}

		r.refsErr = gitErr
		return
	}

	r.refs = make(map[string]string)
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) != 2 {
			continue
		}
		if f[1] == "HEAD" || strings.HasPrefix(f[1], "refs/heads/") || strings.HasPrefix(f[1], "refs/tags/") {
			r.refs[f[1]] = f[0]
		}
	}
	for ref, hash := range r.refs {
		if strings.HasSuffix(ref, "^{}") { // record unwrapped annotated tag as value of tag
			r.refs[strings.TrimSuffix(ref, "^{}")] = hash
			delete(r.refs, ref)
		}
	}
}

func (r *gitRepo) Tags(prefix string) ([]string, error) {
	r.refsOnce.Do(r.loadRefs)
	if r.refsErr != nil {
		return nil, r.refsErr
	}

	tags := []string{}
	for ref := range r.refs {
		if !strings.HasPrefix(ref, "refs/tags/") {
			continue
		}
		tag := ref[len("refs/tags/"):]
		if !strings.HasPrefix(tag, prefix) {
			continue
		}
		tags = append(tags, tag)
	}
	sort.Strings(tags)
	return tags, nil
}

func (r *gitRepo) Latest() (*RevInfo, error) {
	r.refsOnce.Do(r.loadRefs)
	if r.refsErr != nil {
		return nil, r.refsErr
	}
	if r.refs["HEAD"] == "" {
		return nil, ErrNoCommits
	}
	return r.Stat(r.refs["HEAD"])
}

// findRef finds some ref name for the given hash,
// for use when the server requires giving a ref instead of a hash.
// There may be multiple ref names for a given hash,
// in which case this returns some name - it doesn't matter which.
func (r *gitRepo) findRef(hash string) (ref string, ok bool) {
	r.refsOnce.Do(r.loadRefs)
	for ref, h := range r.refs {
		if h == hash {
			return ref, true
		}
	}
	return "", false
}

// minHashDigits is the minimum number of digits to require
// before accepting a hex digit sequence as potentially identifying
// a specific commit in a git repo. (Of course, users can always
// specify more digits, and many will paste in all 40 digits,
// but many of git's commands default to printing short hashes
// as 7 digits.)
const minHashDigits = 7

// stat stats the given rev in the local repository,
// or else it fetches more info from the remote repository and tries again.
func (r *gitRepo) stat(rev string) (*RevInfo, error) {
	if r.local {
		return r.statLocal(rev, rev)
	}

	// Fast path: maybe rev is a hash we already have locally.
	didStatLocal := false
	if len(rev) >= minHashDigits && len(rev) <= 40 && AllHex(rev) {
		if info, err := r.statLocal(rev, rev); err == nil {
			return info, nil
		}
		didStatLocal = true
	}

	// Maybe rev is a tag we already have locally.
	// (Note that we're excluding branches, which can be stale.)
	r.localTagsOnce.Do(r.loadLocalTags)
	if r.localTags[rev] {
		return r.statLocal(rev, "refs/tags/"+rev)
	}

	// Maybe rev is the name of a tag or branch on the remote server.
	// Or maybe it's the prefix of a hash of a named ref.
	// Try to resolve to both a ref (git name) and full (40-hex-digit) commit hash.
	r.refsOnce.Do(r.loadRefs)
	// loadRefs may return an error if git fails, for example segfaults, or
	// could not load a private repo, but defer checking to the else block
	// below, in case we already have the rev in question in the local cache.
	var ref, hash string
	if r.refs["refs/tags/"+rev] != "" {
		ref = "refs/tags/" + rev
		hash = r.refs[ref]
		// Keep rev as is: tags are assumed not to change meaning.
	} else if r.refs["refs/heads/"+rev] != "" {
		ref = "refs/heads/" + rev
		hash = r.refs[ref]
		rev = hash // Replace rev, because meaning of refs/heads/foo can change.
	} else if rev == "HEAD" && r.refs["HEAD"] != "" {
		ref = "HEAD"
		hash = r.refs[ref]
		rev = hash // Replace rev, because meaning of HEAD can change.
	} else if len(rev) >= minHashDigits && len(rev) <= 40 && AllHex(rev) {
		// At the least, we have a hash prefix we can look up after the fetch below.
		// Maybe we can map it to a full hash using the known refs.
		prefix := rev
		// Check whether rev is prefix of known ref hash.
		for k, h := range r.refs {
			if strings.HasPrefix(h, prefix) {
				if hash != "" && hash != h {
					// Hash is an ambiguous hash prefix.
					// More information will not change that.
					return nil, fmt.Errorf("ambiguous revision %s", rev)
				}
				if ref == "" || ref > k { // Break ties deterministically when multiple refs point at same hash.
					ref = k
				}
				rev = h
				hash = h
			}
		}
		if hash == "" && len(rev) == 40 { // Didn't find a ref, but rev is a full hash.
			hash = rev
		}
	} else {
		if r.refsErr != nil {
			return nil, r.refsErr
		}
		return nil, &UnknownRevisionError{Rev: rev}
	}

	// Protect r.fetchLevel and the "fetch more and more" sequence.
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	// Perhaps r.localTags did not have the ref when we loaded local tags,
	// but we've since done fetches that pulled down the hash we need
	// (or already have the hash we need, just without its tag).
	// Either way, try a local stat before falling back to network I/O.
	if !didStatLocal {
		if info, err := r.statLocal(rev, hash); err == nil {
			if strings.HasPrefix(ref, "refs/tags/") {
				// Make sure tag exists, so it will be in localTags next time the go command is run.
				Run(r.dir, "git", "tag", strings.TrimPrefix(ref, "refs/tags/"), hash)
			}
			return info, nil
		}
	}

	// If we know a specific commit we need and its ref, fetch it.
	// We do NOT fetch arbitrary hashes (when we don't know the ref)
	// because we want to avoid ever importing a commit that isn't
	// reachable from refs/tags/* or refs/heads/* or HEAD.
	// Both Gerrit and GitHub expose every CL/PR as a named ref,
	// and we don't want those commits masquerading as being real
	// pseudo-versions in the main repo.
	if r.fetchLevel <= fetchSome && ref != "" && hash != "" && !r.local {
		r.fetchLevel = fetchSome
		var refspec string
		if ref != "" && ref != "HEAD" {
			// If we do know the ref name, save the mapping locally
			// so that (if it is a tag) it can show up in localTags
			// on a future call. Also, some servers refuse to allow
			// full hashes in ref specs, so prefer a ref name if known.
			refspec = ref + ":" + ref
		} else {
			// Fetch the hash but give it a local name (refs/dummy),
			// because that triggers the fetch behavior of creating any
			// other known remote tags for the hash. We never use
			// refs/dummy (it's not refs/tags/dummy) and it will be
			// overwritten in the next command, and that's fine.
			ref = hash
			refspec = hash + ":refs/dummy"
		}
		_, err := Run(r.dir, "git", "fetch", "-f", "--depth=1", r.remote, refspec)
		if err == nil {
			return r.statLocal(rev, ref)
		}
		// Don't try to be smart about parsing the error.
		// It's too complex and varies too much by git version.
		// No matter what went wrong, fall back to a complete fetch.
	}

	// Last resort.
	// Fetch all heads and tags and hope the hash we want is in the history.
	if err := r.fetchRefsLocked(); err != nil {
		return nil, err
	}

	return r.statLocal(rev, rev)
}

// fetchRefsLocked fetches all heads and tags from the origin, along with the
// ancestors of those commits.
//
// We only fetch heads and tags, not arbitrary other commits: we don't want to
// pull in off-branch commits (such as rejected GitHub pull requests) that the
// server may be willing to provide. (See the comments within the stat method
// for more detail.)
//
// fetchRefsLocked requires that r.mu remain locked for the duration of the call.
func (r *gitRepo) fetchRefsLocked() error {
	if r.fetchLevel < fetchAll {
		// NOTE: To work around a bug affecting Git clients up to at least 2.23.0
		// (2019-08-16), we must first expand the set of local refs, and only then
		// unshallow the repository as a separate fetch operation. (See
		// golang.org/issue/34266 and
		// https://github.com/git/git/blob/4c86140027f4a0d2caaa3ab4bd8bfc5ce3c11c8a/transport.c#L1303-L1309.)

		if _, err := Run(r.dir, "git", "fetch", "-f", r.remote, "refs/heads/*:refs/heads/*", "refs/tags/*:refs/tags/*"); err != nil {
			return err
		}

		if _, err := os.Stat(filepath.Join(r.dir, "shallow")); err == nil {
			if _, err := Run(r.dir, "git", "fetch", "--unshallow", "-f", r.remote); err != nil {
				return err
			}
		}

		r.fetchLevel = fetchAll
	}
	return nil
}

// statLocal returns a RevInfo describing rev in the local git repository.
// It uses version as info.Version.
func (r *gitRepo) statLocal(version, rev string) (*RevInfo, error) {
	out, err := Run(r.dir, "git", "-c", "log.showsignature=false", "log", "-n1", "--format=format:%H %ct %D", rev, "--")
	if err != nil {
		return nil, &UnknownRevisionError{Rev: rev}
	}
	f := strings.Fields(string(out))
	if len(f) < 2 {
		return nil, fmt.Errorf("unexpected response from git log: %q", out)
	}
	hash := f[0]
	if strings.HasPrefix(hash, version) {
		version = hash // extend to full hash
	}
	t, err := strconv.ParseInt(f[1], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid time from git log: %q", out)
	}

	info := &RevInfo{
		Name:    hash,
		Short:   ShortenSHA1(hash),
		Time:    time.Unix(t, 0).UTC(),
		Version: hash,
	}

	// Add tags. Output looks like:
	//	ede458df7cd0fdca520df19a33158086a8a68e81 1523994202 HEAD -> master, tag: v1.2.4-annotated, tag: v1.2.3, origin/master, origin/HEAD
	for i := 2; i < len(f); i++ {
		if f[i] == "tag:" {
			i++
			if i < len(f) {
				info.Tags = append(info.Tags, strings.TrimSuffix(f[i], ","))
			}
		}
	}
	sort.Strings(info.Tags)

	// Used hash as info.Version above.
	// Use caller's suggested version if it appears in the tag list
	// (filters out branch names, HEAD).
	for _, tag := range info.Tags {
		if version == tag {
			info.Version = version
		}
	}

	return info, nil
}

func (r *gitRepo) Stat(rev string) (*RevInfo, error) {
	if rev == "latest" {
		return r.Latest()
	}
	type cached struct {
		info *RevInfo
		err  error
	}
	c := r.statCache.Do(rev, func() interface{} {
		info, err := r.stat(rev)
		return cached{info, err}
	}).(cached)
	return c.info, c.err
}

func (r *gitRepo) ReadFile(rev, file string, maxSize int64) ([]byte, error) {
	// TODO: Could use git cat-file --batch.
	info, err := r.Stat(rev) // download rev into local git repo
	if err != nil {
		return nil, err
	}
	out, err := Run(r.dir, "git", "cat-file", "blob", info.Name+":"+file)
	if err != nil {
		return nil, fs.ErrNotExist
	}
	return out, nil
}

func (r *gitRepo) ReadFileRevs(revs []string, file string, maxSize int64) (map[string]*FileRev, error) {
	// Create space to hold results.
	files := make(map[string]*FileRev)
	for _, rev := range revs {
		f := &FileRev{Rev: rev}
		files[rev] = f
	}

	// Collect locally-known revs.
	need, err := r.readFileRevs(revs, file, files)
	if err != nil {
		return nil, err
	}
	if len(need) == 0 {
		return files, nil
	}

	// Build list of known remote refs that might help.
	var redo []string
	r.refsOnce.Do(r.loadRefs)
	if r.refsErr != nil {
		return nil, r.refsErr
	}
	for _, tag := range need {
		if r.refs["refs/tags/"+tag] != "" {
			redo = append(redo, tag)
		}
	}
	if len(redo) == 0 {
		return files, nil
	}

	// Protect r.fetchLevel and the "fetch more and more" sequence.
	// See stat method above.
	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	if err := r.fetchRefsLocked(); err != nil {
		return nil, err
	}

	if _, err := r.readFileRevs(redo, file, files); err != nil {
		return nil, err
	}

	return files, nil
}

func (r *gitRepo) readFileRevs(tags []string, file string, fileMap map[string]*FileRev) (missing []string, err error) {
	var stdin bytes.Buffer
	for _, tag := range tags {
		fmt.Fprintf(&stdin, "refs/tags/%s\n", tag)
		fmt.Fprintf(&stdin, "refs/tags/%s:%s\n", tag, file)
	}

	data, err := RunWithStdin(r.dir, &stdin, "git", "cat-file", "--batch")
	if err != nil {
		return nil, err
	}

	next := func() (typ string, body []byte, ok bool) {
		var line string
		i := bytes.IndexByte(data, '\n')
		if i < 0 {
			return "", nil, false
		}
		line, data = string(bytes.TrimSpace(data[:i])), data[i+1:]
		if strings.HasSuffix(line, " missing") {
			return "missing", nil, true
		}
		f := strings.Fields(line)
		if len(f) != 3 {
			return "", nil, false
		}
		n, err := strconv.Atoi(f[2])
		if err != nil || n > len(data) {
			return "", nil, false
		}
		body, data = data[:n], data[n:]
		if len(data) > 0 && data[0] == '\r' {
			data = data[1:]
		}
		if len(data) > 0 && data[0] == '\n' {
			data = data[1:]
		}
		return f[1], body, true
	}

	badGit := func() ([]string, error) {
		return nil, fmt.Errorf("malformed output from git cat-file --batch")
	}

	for _, tag := range tags {
		commitType, _, ok := next()
		if !ok {
			return badGit()
		}
		fileType, fileData, ok := next()
		if !ok {
			return badGit()
		}
		f := fileMap[tag]
		f.Data = nil
		f.Err = nil
		switch commitType {
		default:
			f.Err = fmt.Errorf("unexpected non-commit type %q for rev %s", commitType, tag)

		case "missing":
			// Note: f.Err must not satisfy os.IsNotExist. That's reserved for the file not existing in a valid commit.
			f.Err = fmt.Errorf("no such rev %s", tag)
			missing = append(missing, tag)

		case "tag", "commit":
			switch fileType {
			default:
				f.Err = &fs.PathError{Path: tag + ":" + file, Op: "read", Err: fmt.Errorf("unexpected non-blob type %q", fileType)}
			case "missing":
				f.Err = &fs.PathError{Path: tag + ":" + file, Op: "read", Err: fs.ErrNotExist}
			case "blob":
				f.Data = fileData
			}
		}
	}
	if len(bytes.TrimSpace(data)) != 0 {
		return badGit()
	}

	return missing, nil
}

func (r *gitRepo) RecentTag(rev, prefix string, allowed func(string) bool) (tag string, err error) {
	info, err := r.Stat(rev)
	if err != nil {
		return "", err
	}
	rev = info.Name // expand hash prefixes

	// describe sets tag and err using 'git for-each-ref' and reports whether the
	// result is definitive.
	describe := func() (definitive bool) {
		var out []byte
		out, err = Run(r.dir, "git", "for-each-ref", "--format", "%(refname)", "refs/tags", "--merged", rev)
		if err != nil {
			return true
		}

		// prefixed tags aren't valid semver tags so compare without prefix, but only tags with correct prefix
		var highest string
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			// git do support lstrip in for-each-ref format, but it was added in v2.13.0. Stripping here
			// instead gives support for git v2.7.0.
			if !strings.HasPrefix(line, "refs/tags/") {
				continue
			}
			line = line[len("refs/tags/"):]

			if !strings.HasPrefix(line, prefix) {
				continue
			}

			semtag := line[len(prefix):]
			// Consider only tags that are valid and complete (not just major.minor prefixes).
			// NOTE: Do not replace the call to semver.Compare with semver.Max.
			// We want to return the actual tag, not a canonicalized version of it,
			// and semver.Max currently canonicalizes (see golang.org/issue/32700).
			if c := semver.Canonical(semtag); c == "" || !strings.HasPrefix(semtag, c) || !allowed(semtag) {
				continue
			}
			if semver.Compare(semtag, highest) > 0 {
				highest = semtag
			}
		}

		if highest != "" {
			tag = prefix + highest
		}

		return tag != "" && !AllHex(tag)
	}

	if describe() {
		return tag, err
	}

	// Git didn't find a version tag preceding the requested rev.
	// See whether any plausible tag exists.
	tags, err := r.Tags(prefix + "v")
	if err != nil {
		return "", err
	}
	if len(tags) == 0 {
		return "", nil
	}

	// There are plausible tags, but we don't know if rev is a descendent of any of them.
	// Fetch the history to find out.

	unlock, err := r.mu.Lock()
	if err != nil {
		return "", err
	}
	defer unlock()

	if err := r.fetchRefsLocked(); err != nil {
		return "", err
	}

	// If we've reached this point, we have all of the commits that are reachable
	// from all heads and tags.
	//
	// The only refs we should be missing are those that are no longer reachable
	// (or never were reachable) from any branch or tag, including the master
	// branch, and we don't want to resolve them anyway (they're probably
	// unreachable for a reason).
	//
	// Try one last time in case some other goroutine fetched rev while we were
	// waiting on the lock.
	describe()
	return tag, err
}

func (r *gitRepo) DescendsFrom(rev, tag string) (bool, error) {
	// The "--is-ancestor" flag was added to "git merge-base" in version 1.8.0, so
	// this won't work with Git 1.7.1. According to golang.org/issue/28550, cmd/go
	// already doesn't work with Git 1.7.1, so at least it's not a regression.
	//
	// git merge-base --is-ancestor exits with status 0 if rev is an ancestor, or
	// 1 if not.
	_, err := Run(r.dir, "git", "merge-base", "--is-ancestor", "--", tag, rev)

	// Git reports "is an ancestor" with exit code 0 and "not an ancestor" with
	// exit code 1.
	// Unfortunately, if we've already fetched rev with a shallow history, git
	// merge-base has been observed to report a false-negative, so don't stop yet
	// even if the exit code is 1!
	if err == nil {
		return true, nil
	}

	// See whether the tag and rev even exist.
	tags, err := r.Tags(tag)
	if err != nil {
		return false, err
	}
	if len(tags) == 0 {
		return false, nil
	}

	// NOTE: r.stat is very careful not to fetch commits that we shouldn't know
	// about, like rejected GitHub pull requests, so don't try to short-circuit
	// that here.
	if _, err = r.stat(rev); err != nil {
		return false, err
	}

	// Now fetch history so that git can search for a path.
	unlock, err := r.mu.Lock()
	if err != nil {
		return false, err
	}
	defer unlock()

	if r.fetchLevel < fetchAll {
		// Fetch the complete history for all refs and heads. It would be more
		// efficient to only fetch the history from rev to tag, but that's much more
		// complicated, and any kind of shallow fetch is fairly likely to trigger
		// bugs in JGit servers and/or the go command anyway.
		if err := r.fetchRefsLocked(); err != nil {
			return false, err
		}
	}

	_, err = Run(r.dir, "git", "merge-base", "--is-ancestor", "--", tag, rev)
	if err == nil {
		return true, nil
	}
	if ee, ok := err.(*RunError).Err.(*exec.ExitError); ok && ee.ExitCode() == 1 {
		return false, nil
	}
	return false, err
}

func (r *gitRepo) ReadZip(rev, subdir string, maxSize int64) (zip io.ReadCloser, err error) {
	// TODO: Use maxSize or drop it.
	args := []string{}
	if subdir != "" {
		args = append(args, "--", subdir)
	}
	info, err := r.Stat(rev) // download rev into local git repo
	if err != nil {
		return nil, err
	}

	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	if err := ensureGitAttributes(r.dir); err != nil {
		return nil, err
	}

	// Incredibly, git produces different archives depending on whether
	// it is running on a Windows system or not, in an attempt to normalize
	// text file line endings. Setting -c core.autocrlf=input means only
	// translate files on the way into the repo, not on the way out (archive).
	// The -c core.eol=lf should be unnecessary but set it anyway.
	archive, err := Run(r.dir, "git", "-c", "core.autocrlf=input", "-c", "core.eol=lf", "archive", "--format=zip", "--prefix=prefix/", info.Name, args)
	if err != nil {
		if bytes.Contains(err.(*RunError).Stderr, []byte("did not match any files")) {
			return nil, fs.ErrNotExist
		}
		return nil, err
	}

	return io.NopCloser(bytes.NewReader(archive)), nil
}

// ensureGitAttributes makes sure export-subst and export-ignore features are
// disabled for this repo. This is intended to be run prior to running git
// archive so that zip files are generated that produce consistent ziphashes
// for a given revision, independent of variables such as git version and the
// size of the repo.
//
// See: https://github.com/golang/go/issues/27153
func ensureGitAttributes(repoDir string) (err error) {
	const attr = "\n* -export-subst -export-ignore\n"

	d := repoDir + "/info"
	p := d + "/attributes"

	if err := os.MkdirAll(d, 0755); err != nil {
		return err
	}

	f, err := os.OpenFile(p, os.O_CREATE|os.O_APPEND|os.O_RDWR, 0666)
	if err != nil {
		return err
	}
	defer func() {
		closeErr := f.Close()
		if closeErr != nil {
			err = closeErr
		}
	}()

	b, err := io.ReadAll(f)
	if err != nil {
		return err
	}
	if !bytes.HasSuffix(b, []byte(attr)) {
		_, err := f.WriteString(attr)
		return err
	}

	return nil
}
