// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/web"
	"cmd/internal/par"

	"golang.org/x/mod/semver"
)

// A notExistError wraps another error to retain its original text
// but makes it opaquely equivalent to fs.ErrNotExist.
type notExistError struct {
	err error
}

func (e notExistError) Error() string   { return e.err.Error() }
func (notExistError) Is(err error) bool { return err == fs.ErrNotExist }

const gitWorkDirType = "git3"

func newGitRepo(ctx context.Context, remote string, local bool) (Repo, error) {
	r := &gitRepo{remote: remote, local: local}
	if local {
		if strings.Contains(remote, "://") { // Local flag, but URL provided
			return nil, fmt.Errorf("git remote (%s) lookup disabled", remote)
		}
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
	// This is a remote path lookup.
	if !strings.Contains(remote, "://") { // No URL scheme, could be host:path
		if strings.Contains(remote, ":") {
			return nil, fmt.Errorf("git remote (%s) must not be local directory (use URL syntax not host:path syntax)", remote)
		}
		return nil, fmt.Errorf("git remote (%s) must not be local directory", remote)
	}
	var err error
	r.dir, r.mu.Path, err = WorkDir(ctx, gitWorkDirType, r.remote)
	if err != nil {
		return nil, err
	}

	unlock, err := r.mu.Lock()
	if err != nil {
		return nil, err
	}
	defer unlock()

	if _, err := os.Stat(filepath.Join(r.dir, "objects")); err != nil {
		if _, err := Run(ctx, r.dir, "git", "init", "--bare"); err != nil {
			os.RemoveAll(r.dir)
			return nil, err
		}
		// We could just say git fetch https://whatever later,
		// but this lets us say git fetch origin instead, which
		// is a little nicer. More importantly, using a named remote
		// avoids a problem with Git LFS. See golang.org/issue/25605.
		if _, err := r.runGit(ctx, "git", "remote", "add", "origin", "--", r.remote); err != nil {
			os.RemoveAll(r.dir)
			return nil, err
		}
		if runtime.GOOS == "windows" {
			// Git for Windows by default does not support paths longer than
			// MAX_PATH (260 characters) because that may interfere with navigation
			// in some Windows programs. However, cmd/go should be able to handle
			// long paths just fine, and we expect people to use 'go clean' to
			// manipulate the module cache, so it should be harmless to set here,
			// and in some cases may be necessary in order to download modules with
			// long branch names.
			//
			// See https://github.com/git-for-windows/git/wiki/Git-cannot-create-a-file-or-directory-with-a-long-path.
			if _, err := r.runGit(ctx, "git", "config", "core.longpaths", "true"); err != nil {
				os.RemoveAll(r.dir)
				return nil, err
			}
		}
	}
	r.remoteURL = r.remote
	r.remote = "origin"
	return r, nil
}

type gitRepo struct {
	ctx context.Context

	remote, remoteURL string
	local             bool
	dir               string

	mu lockedfile.Mutex // protects fetchLevel and git repo state

	fetchLevel int

	statCache par.ErrCache[string, *RevInfo]

	refsOnce sync.Once
	// refs maps branch and tag refs (e.g., "HEAD", "refs/heads/master")
	// to commits (e.g., "37ffd2e798afde829a34e8955b716ab730b2a6d6")
	refs    map[string]string
	refsErr error

	localTagsOnce sync.Once
	localTags     sync.Map // map[string]bool
}

const (
	// How much have we fetched into the git repo (in this process)?
	fetchNone = iota // nothing yet
	fetchSome        // shallow fetches of individual hashes
	fetchAll         // "fetch -t origin": get all remote branches and tags
)

// loadLocalTags loads tag references from the local git cache
// into the map r.localTags.
func (r *gitRepo) loadLocalTags(ctx context.Context) {
	// The git protocol sends all known refs and ls-remote filters them on the client side,
	// so we might as well record both heads and tags in one shot.
	// Most of the time we only care about tags but sometimes we care about heads too.
	out, err := r.runGit(ctx, "git", "tag", "-l")
	if err != nil {
		return
	}

	for _, line := range strings.Split(string(out), "\n") {
		if line != "" {
			r.localTags.Store(line, true)
		}
	}
}

func (r *gitRepo) CheckReuse(ctx context.Context, old *Origin, subdir string) error {
	if old == nil {
		return fmt.Errorf("missing origin")
	}
	if old.VCS != "git" || old.URL != r.remoteURL {
		return fmt.Errorf("origin moved from %v %q to %v %q", old.VCS, old.URL, "git", r.remoteURL)
	}
	if old.Subdir != subdir {
		return fmt.Errorf("origin moved from %v %q %q to %v %q %q", old.VCS, old.URL, old.Subdir, "git", r.remoteURL, subdir)
	}

	// Note: Can have Hash with no Ref and no TagSum and no RepoSum,
	// meaning the Hash simply has to remain in the repo.
	// In that case we assume it does in the absence of any real way to check.
	// But if neither Hash nor TagSum is present, we have nothing to check,
	// which we take to mean we didn't record enough information to be sure.
	if old.Hash == "" && old.TagSum == "" && old.RepoSum == "" {
		return fmt.Errorf("non-specific origin")
	}

	r.loadRefs(ctx)
	if r.refsErr != nil {
		return r.refsErr
	}

	if old.Ref != "" {
		hash, ok := r.refs[old.Ref]
		if !ok {
			return fmt.Errorf("ref %q deleted", old.Ref)
		}
		if hash != old.Hash {
			return fmt.Errorf("ref %q moved from %s to %s", old.Ref, old.Hash, hash)
		}
	}
	if old.TagSum != "" {
		tags, err := r.Tags(ctx, old.TagPrefix)
		if err != nil {
			return err
		}
		if tags.Origin.TagSum != old.TagSum {
			return fmt.Errorf("tags changed")
		}
	}
	if old.RepoSum != "" {
		if r.repoSum(r.refs) != old.RepoSum {
			return fmt.Errorf("refs changed")
		}
	}
	return nil
}

// loadRefs loads heads and tags references from the remote into the map r.refs.
// The result is cached in memory.
func (r *gitRepo) loadRefs(ctx context.Context) (map[string]string, error) {
	r.refsOnce.Do(func() {
		// The git protocol sends all known refs and ls-remote filters them on the client side,
		// so we might as well record both heads and tags in one shot.
		// Most of the time we only care about tags but sometimes we care about heads too.
		release, err := base.AcquireNet()
		if err != nil {
			r.refsErr = err
			return
		}
		out, gitErr := r.runGit(ctx, "git", "ls-remote", "-q", r.remote)
		release()

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

		refs := make(map[string]string)
		for _, line := range strings.Split(string(out), "\n") {
			f := strings.Fields(line)
			if len(f) != 2 {
				continue
			}
			if f[1] == "HEAD" || strings.HasPrefix(f[1], "refs/heads/") || strings.HasPrefix(f[1], "refs/tags/") {
				refs[f[1]] = f[0]
			}
		}
		for ref, hash := range refs {
			if k, found := strings.CutSuffix(ref, "^{}"); found { // record unwrapped annotated tag as value of tag
				refs[k] = hash
				delete(refs, ref)
			}
		}
		r.refs = refs
	})
	return r.refs, r.refsErr
}

func (r *gitRepo) Tags(ctx context.Context, prefix string) (*Tags, error) {
	refs, err := r.loadRefs(ctx)
	if err != nil {
		return nil, err
	}

	tags := &Tags{
		Origin: &Origin{
			VCS:       "git",
			URL:       r.remoteURL,
			TagPrefix: prefix,
		},
		List: []Tag{},
	}
	for ref, hash := range refs {
		if !strings.HasPrefix(ref, "refs/tags/") {
			continue
		}
		tag := ref[len("refs/tags/"):]
		if !strings.HasPrefix(tag, prefix) {
			continue
		}
		tags.List = append(tags.List, Tag{tag, hash})
	}
	sort.Slice(tags.List, func(i, j int) bool {
		return tags.List[i].Name < tags.List[j].Name
	})

	dir := prefix[:strings.LastIndex(prefix, "/")+1]
	h := sha256.New()
	for _, tag := range tags.List {
		if isOriginTag(strings.TrimPrefix(tag.Name, dir)) {
			fmt.Fprintf(h, "%q %s\n", tag.Name, tag.Hash)
		}
	}
	tags.Origin.TagSum = "t1:" + base64.StdEncoding.EncodeToString(h.Sum(nil))
	return tags, nil
}

// repoSum returns a checksum of the entire repo state,
// which can be checked (as Origin.RepoSum) to cache
// the absence of a specific module version.
// The caller must supply refs, the result of a successful r.loadRefs.
func (r *gitRepo) repoSum(refs map[string]string) string {
	var list []string
	for ref := range refs {
		list = append(list, ref)
	}
	sort.Strings(list)
	h := sha256.New()
	for _, ref := range list {
		fmt.Fprintf(h, "%q %s\n", ref, refs[ref])
	}
	return "r1:" + base64.StdEncoding.EncodeToString(h.Sum(nil))
}

// unknownRevisionInfo returns a RevInfo containing an Origin containing a RepoSum of refs,
// for use when returning an UnknownRevisionError.
func (r *gitRepo) unknownRevisionInfo(refs map[string]string) *RevInfo {
	return &RevInfo{
		Origin: &Origin{
			VCS:     "git",
			URL:     r.remoteURL,
			RepoSum: r.repoSum(refs),
		},
	}
}

func (r *gitRepo) Latest(ctx context.Context) (*RevInfo, error) {
	refs, err := r.loadRefs(ctx)
	if err != nil {
		return nil, err
	}
	if refs["HEAD"] == "" {
		return nil, ErrNoCommits
	}
	statInfo, err := r.Stat(ctx, refs["HEAD"])
	if err != nil {
		return nil, err
	}

	// Stat may return cached info, so make a copy to modify here.
	info := new(RevInfo)
	*info = *statInfo
	info.Origin = new(Origin)
	if statInfo.Origin != nil {
		*info.Origin = *statInfo.Origin
	}
	info.Origin.Ref = "HEAD"
	info.Origin.Hash = refs["HEAD"]

	return info, nil
}

// findRef finds some ref name for the given hash,
// for use when the server requires giving a ref instead of a hash.
// There may be multiple ref names for a given hash,
// in which case this returns some name - it doesn't matter which.
func (r *gitRepo) findRef(ctx context.Context, hash string) (ref string, ok bool) {
	refs, err := r.loadRefs(ctx)
	if err != nil {
		return "", false
	}
	for ref, h := range refs {
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
func (r *gitRepo) stat(ctx context.Context, rev string) (info *RevInfo, err error) {
	if r.local {
		return r.statLocal(ctx, rev, rev)
	}

	// Fast path: maybe rev is a hash we already have locally.
	didStatLocal := false
	if len(rev) >= minHashDigits && len(rev) <= 40 && AllHex(rev) {
		if info, err := r.statLocal(ctx, rev, rev); err == nil {
			return info, nil
		}
		didStatLocal = true
	}

	// Maybe rev is a tag we already have locally.
	// (Note that we're excluding branches, which can be stale.)
	r.localTagsOnce.Do(func() { r.loadLocalTags(ctx) })
	if _, ok := r.localTags.Load(rev); ok {
		return r.statLocal(ctx, rev, "refs/tags/"+rev)
	}

	// Maybe rev is the name of a tag or branch on the remote server.
	// Or maybe it's the prefix of a hash of a named ref.
	// Try to resolve to both a ref (git name) and full (40-hex-digit) commit hash.
	refs, err := r.loadRefs(ctx)
	if err != nil {
		return nil, err
	}
	// loadRefs may return an error if git fails, for example segfaults, or
	// could not load a private repo, but defer checking to the else block
	// below, in case we already have the rev in question in the local cache.
	var ref, hash string
	if refs["refs/tags/"+rev] != "" {
		ref = "refs/tags/" + rev
		hash = refs[ref]
		// Keep rev as is: tags are assumed not to change meaning.
	} else if refs["refs/heads/"+rev] != "" {
		ref = "refs/heads/" + rev
		hash = refs[ref]
		rev = hash // Replace rev, because meaning of refs/heads/foo can change.
	} else if rev == "HEAD" && refs["HEAD"] != "" {
		ref = "HEAD"
		hash = refs[ref]
		rev = hash // Replace rev, because meaning of HEAD can change.
	} else if len(rev) >= minHashDigits && len(rev) <= 40 && AllHex(rev) {
		// At the least, we have a hash prefix we can look up after the fetch below.
		// Maybe we can map it to a full hash using the known refs.
		prefix := rev
		// Check whether rev is prefix of known ref hash.
		for k, h := range refs {
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
		return r.unknownRevisionInfo(refs), &UnknownRevisionError{Rev: rev}
	}

	defer func() {
		if info != nil {
			info.Origin.Hash = info.Name
			// There's a ref = hash below; don't write that hash down as Origin.Ref.
			if ref != info.Origin.Hash {
				info.Origin.Ref = ref
			}
		}
	}()

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
		if info, err := r.statLocal(ctx, rev, hash); err == nil {
			tag, fromTag := strings.CutPrefix(ref, "refs/tags/")
			if fromTag && !slices.Contains(info.Tags, tag) {
				// The local repo includes the commit hash we want, but it is missing
				// the corresponding tag. Add that tag and try again.
				_, err := r.runGit(ctx, "git", "tag", tag, hash)
				if err != nil {
					return nil, err
				}
				r.localTags.Store(tag, true)
				return r.statLocal(ctx, rev, ref)
			}
			return info, err
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
		if ref == "HEAD" {
			// Fetch the hash but give it a local name (refs/dummy),
			// because that triggers the fetch behavior of creating any
			// other known remote tags for the hash. We never use
			// refs/dummy (it's not refs/tags/dummy) and it will be
			// overwritten in the next command, and that's fine.
			ref = hash
			refspec = hash + ":refs/dummy"
		} else {
			// If we do know the ref name, save the mapping locally
			// so that (if it is a tag) it can show up in localTags
			// on a future call. Also, some servers refuse to allow
			// full hashes in ref specs, so prefer a ref name if known.
			refspec = ref + ":" + ref
		}

		release, err := base.AcquireNet()
		if err != nil {
			return nil, err
		}
		// We explicitly set protocol.version=2 for this command to work around
		// an apparent Git bug introduced in Git 2.21 (commit 61c771),
		// which causes the handler for protocol version 1 to sometimes miss
		// tags that point to the requested commit (see https://go.dev/issue/56881).
		_, err = r.runGit(ctx, "git", "-c", "protocol.version=2", "fetch", "-f", "--depth=1", r.remote, refspec)
		release()

		if err == nil {
			return r.statLocal(ctx, rev, ref)
		}
		// Don't try to be smart about parsing the error.
		// It's too complex and varies too much by git version.
		// No matter what went wrong, fall back to a complete fetch.
	}

	// Last resort.
	// Fetch all heads and tags and hope the hash we want is in the history.
	if err := r.fetchRefsLocked(ctx); err != nil {
		return nil, err
	}

	return r.statLocal(ctx, rev, rev)
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
func (r *gitRepo) fetchRefsLocked(ctx context.Context) error {
	if r.fetchLevel < fetchAll {
		// NOTE: To work around a bug affecting Git clients up to at least 2.23.0
		// (2019-08-16), we must first expand the set of local refs, and only then
		// unshallow the repository as a separate fetch operation. (See
		// golang.org/issue/34266 and
		// https://github.com/git/git/blob/4c86140027f4a0d2caaa3ab4bd8bfc5ce3c11c8a/transport.c#L1303-L1309.)

		release, err := base.AcquireNet()
		if err != nil {
			return err
		}
		defer release()

		if _, err := r.runGit(ctx, "git", "fetch", "-f", r.remote, "refs/heads/*:refs/heads/*", "refs/tags/*:refs/tags/*"); err != nil {
			return err
		}

		if _, err := os.Stat(filepath.Join(r.dir, "shallow")); err == nil {
			if _, err := r.runGit(ctx, "git", "fetch", "--unshallow", "-f", r.remote); err != nil {
				return err
			}
		}

		r.fetchLevel = fetchAll
	}
	return nil
}

// statLocal returns a new RevInfo describing rev in the local git repository.
// It uses version as info.Version.
func (r *gitRepo) statLocal(ctx context.Context, version, rev string) (*RevInfo, error) {
	out, err := r.runGit(ctx, "git", "-c", "log.showsignature=false", "log", "--no-decorate", "-n1", "--format=format:%H %ct %D", rev, "--")
	if err != nil {
		// Return info with Origin.RepoSum if possible to allow caching of negative lookup.
		var info *RevInfo
		if refs, err := r.loadRefs(ctx); err == nil {
			info = r.unknownRevisionInfo(refs)
		}
		return info, &UnknownRevisionError{Rev: rev}
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
		Origin: &Origin{
			VCS:  "git",
			URL:  r.remoteURL,
			Hash: hash,
		},
		Name:    hash,
		Short:   ShortenSHA1(hash),
		Time:    time.Unix(t, 0).UTC(),
		Version: hash,
	}
	if !strings.HasPrefix(hash, rev) {
		info.Origin.Ref = rev
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

func (r *gitRepo) Stat(ctx context.Context, rev string) (*RevInfo, error) {
	if rev == "latest" {
		return r.Latest(ctx)
	}
	return r.statCache.Do(rev, func() (*RevInfo, error) {
		return r.stat(ctx, rev)
	})
}

func (r *gitRepo) ReadFile(ctx context.Context, rev, file string, maxSize int64) ([]byte, error) {
	// TODO: Could use git cat-file --batch.
	info, err := r.Stat(ctx, rev) // download rev into local git repo
	if err != nil {
		return nil, err
	}
	out, err := r.runGit(ctx, "git", "cat-file", "blob", info.Name+":"+file)
	if err != nil {
		return nil, fs.ErrNotExist
	}
	return out, nil
}

func (r *gitRepo) RecentTag(ctx context.Context, rev, prefix string, allowed func(tag string) bool) (tag string, err error) {
	info, err := r.Stat(ctx, rev)
	if err != nil {
		return "", err
	}
	rev = info.Name // expand hash prefixes

	// describe sets tag and err using 'git for-each-ref' and reports whether the
	// result is definitive.
	describe := func() (definitive bool) {
		var out []byte
		out, err = r.runGit(ctx, "git", "for-each-ref", "--format", "%(refname)", "refs/tags", "--merged", rev)
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
			if !allowed(line) {
				continue
			}

			semtag := line[len(prefix):]
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
	tags, err := r.Tags(ctx, prefix+"v")
	if err != nil {
		return "", err
	}
	if len(tags.List) == 0 {
		return "", nil
	}

	// There are plausible tags, but we don't know if rev is a descendent of any of them.
	// Fetch the history to find out.

	unlock, err := r.mu.Lock()
	if err != nil {
		return "", err
	}
	defer unlock()

	if err := r.fetchRefsLocked(ctx); err != nil {
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

func (r *gitRepo) DescendsFrom(ctx context.Context, rev, tag string) (bool, error) {
	// The "--is-ancestor" flag was added to "git merge-base" in version 1.8.0, so
	// this won't work with Git 1.7.1. According to golang.org/issue/28550, cmd/go
	// already doesn't work with Git 1.7.1, so at least it's not a regression.
	//
	// git merge-base --is-ancestor exits with status 0 if rev is an ancestor, or
	// 1 if not.
	_, err := r.runGit(ctx, "git", "merge-base", "--is-ancestor", "--", tag, rev)

	// Git reports "is an ancestor" with exit code 0 and "not an ancestor" with
	// exit code 1.
	// Unfortunately, if we've already fetched rev with a shallow history, git
	// merge-base has been observed to report a false-negative, so don't stop yet
	// even if the exit code is 1!
	if err == nil {
		return true, nil
	}

	// See whether the tag and rev even exist.
	tags, err := r.Tags(ctx, tag)
	if err != nil {
		return false, err
	}
	if len(tags.List) == 0 {
		return false, nil
	}

	// NOTE: r.stat is very careful not to fetch commits that we shouldn't know
	// about, like rejected GitHub pull requests, so don't try to short-circuit
	// that here.
	if _, err = r.stat(ctx, rev); err != nil {
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
		if err := r.fetchRefsLocked(ctx); err != nil {
			return false, err
		}
	}

	_, err = r.runGit(ctx, "git", "merge-base", "--is-ancestor", "--", tag, rev)
	if err == nil {
		return true, nil
	}
	if ee, ok := err.(*RunError).Err.(*exec.ExitError); ok && ee.ExitCode() == 1 {
		return false, nil
	}
	return false, err
}

func (r *gitRepo) ReadZip(ctx context.Context, rev, subdir string, maxSize int64) (zip io.ReadCloser, err error) {
	// TODO: Use maxSize or drop it.
	args := []string{}
	if subdir != "" {
		args = append(args, "--", subdir)
	}
	info, err := r.Stat(ctx, rev) // download rev into local git repo
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
	archive, err := r.runGit(ctx, "git", "-c", "core.autocrlf=input", "-c", "core.eol=lf", "archive", "--format=zip", "--prefix=prefix/", info.Name, args)
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

func (r *gitRepo) runGit(ctx context.Context, cmdline ...any) ([]byte, error) {
	args := RunArgs{cmdline: cmdline, dir: r.dir, local: r.local}
	if !r.local {
		// Manually supply GIT_DIR so Git works with safe.bareRepository=explicit set.
		// This is necessary only for remote repositories as they are initialized with git init --bare.
		args.env = []string{"GIT_DIR=" + r.dir}
	}
	return RunWithArgs(ctx, args)
}
