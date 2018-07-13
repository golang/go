// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/par"
)

// GitRepo returns the code repository at the given Git remote reference.
func GitRepo(remote string) (Repo, error) {
	return newGitRepoCached(remote, false)
}

// LocalGitRepo is like Repo but accepts both Git remote references
// and paths to repositories on the local file system.
func LocalGitRepo(remote string) (Repo, error) {
	return newGitRepoCached(remote, true)
}

const gitWorkDirType = "git2"

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
		dir, err := WorkDir(gitWorkDirType, r.remote)
		if err != nil {
			return nil, err
		}
		r.dir = dir
		if _, err := os.Stat(filepath.Join(dir, "objects")); err != nil {
			if _, err := Run(dir, "git", "init", "--bare"); err != nil {
				os.RemoveAll(dir)
				return nil, err
			}
			// We could just say git fetch https://whatever later,
			// but this lets us say git fetch origin instead, which
			// is a little nicer. More importantly, using a named remote
			// avoids a problem with Git LFS. See golang.org/issue/25605.
			if _, err := Run(dir, "git", "remote", "add", "origin", r.remote); err != nil {
				os.RemoveAll(dir)
				return nil, err
			}
			r.remote = "origin"
		}
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
	}
	return r, nil
}

type gitRepo struct {
	remote string
	local  bool
	dir    string

	mu         sync.Mutex // protects fetchLevel, some git repo state
	fetchLevel int

	statCache par.Cache

	refsOnce sync.Once
	refs     map[string]string
	refsErr  error

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
	out, err := Run(r.dir, "git", "ls-remote", "-q", r.remote)
	if err != nil {
		r.refsErr = err
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
		return nil, fmt.Errorf("no commits")
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

func unshallow(gitDir string) []string {
	if _, err := os.Stat(filepath.Join(gitDir, "shallow")); err == nil {
		return []string{"--unshallow"}
	}
	return []string{}
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
	if len(rev) >= minHashDigits && len(rev) <= 40 && AllHex(rev) {
		if info, err := r.statLocal(rev, rev); err == nil {
			return info, nil
		}
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
		return nil, fmt.Errorf("unknown revision %s", rev)
	}

	// Protect r.fetchLevel and the "fetch more and more" sequence.
	// TODO(rsc): Add LockDir and use it for protecting that
	// sequence, so that multiple processes don't collide in their
	// git commands.
	r.mu.Lock()
	defer r.mu.Unlock()

	// If we know a specific commit we need, fetch it.
	if r.fetchLevel <= fetchSome && hash != "" && !r.local {
		r.fetchLevel = fetchSome
		var refspec string
		if ref != "" && ref != "head" {
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
		if !strings.Contains(err.Error(), "unadvertised object") && !strings.Contains(err.Error(), "no such remote ref") && !strings.Contains(err.Error(), "does not support shallow") {
			return nil, err
		}
	}

	// Last resort.
	// Fetch all heads and tags and hope the hash we want is in the history.
	if r.fetchLevel < fetchAll {
		r.fetchLevel = fetchAll

		// To work around a protocol version 2 bug that breaks --unshallow,
		// add -c protocol.version=0.
		// TODO(rsc): The bug is believed to be server-side, meaning only
		// on Google's Git servers. Once the servers are fixed, drop the
		// protocol.version=0. See Google-internal bug b/110495752.
		var protoFlag []string
		unshallowFlag := unshallow(r.dir)
		if len(unshallowFlag) > 0 {
			protoFlag = []string{"-c", "protocol.version=0"}
		}
		if _, err := Run(r.dir, "git", protoFlag, "fetch", unshallowFlag, "-f", r.remote, "refs/heads/*:refs/heads/*", "refs/tags/*:refs/tags/*"); err != nil {
			return nil, err
		}
	}

	return r.statLocal(rev, rev)
}

// statLocal returns a RevInfo describing rev in the local git repository.
// It uses version as info.Version.
func (r *gitRepo) statLocal(version, rev string) (*RevInfo, error) {
	out, err := Run(r.dir, "git", "log", "-n1", "--format=format:%H %ct %D", rev)
	if err != nil {
		return nil, fmt.Errorf("unknown revision %s", rev)
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
		return nil, os.ErrNotExist
	}
	return out, nil
}

func (r *gitRepo) ReadZip(rev, subdir string, maxSize int64) (zip io.ReadCloser, actualSubdir string, err error) {
	// TODO: Use maxSize or drop it.
	args := []string{}
	if subdir != "" {
		args = append(args, "--", subdir)
	}
	info, err := r.Stat(rev) // download rev into local git repo
	if err != nil {
		return nil, "", err
	}

	// Incredibly, git produces different archives depending on whether
	// it is running on a Windows system or not, in an attempt to normalize
	// text file line endings. Setting -c core.autocrlf=input means only
	// translate files on the way into the repo, not on the way out (archive).
	// The -c core.eol=lf should be unnecessary but set it anyway.
	archive, err := Run(r.dir, "git", "-c", "core.autocrlf=input", "-c", "core.eol=lf", "archive", "--format=zip", "--prefix=prefix/", info.Name, args)
	if err != nil {
		if bytes.Contains(err.(*RunError).Stderr, []byte("did not match any files")) {
			return nil, "", os.ErrNotExist
		}
		return nil, "", err
	}

	return ioutil.NopCloser(bytes.NewReader(archive)), "", nil
}
