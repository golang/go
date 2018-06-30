// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gitrepo provides a Git-based implementation of codehost.Repo.
package gitrepo

import (
	"archive/zip"
	"bytes"
	"cmd/go/internal/modfetch/codehost"
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
)

// Repo returns the code repository at the given Git remote reference.
// The returned repo reports the given root as its module root.
func Repo(remote, root string) (codehost.Repo, error) {
	return newRepo(remote, root, false)
}

// LocalRepo is like Repo but accepts both Git remote references
// and paths to repositories on the local file system.
// The returned repo reports the given root as its module root.
func LocalRepo(remote, root string) (codehost.Repo, error) {
	return newRepo(remote, root, true)
}

const workDirType = "git2"

func newRepo(remote, root string, localOK bool) (codehost.Repo, error) {
	r := &repo{remote: remote, root: root, canArchive: true}
	if strings.Contains(remote, "://") {
		// This is a remote path.
		dir, err := codehost.WorkDir(workDirType, r.remote)
		if err != nil {
			return nil, err
		}
		r.dir = dir
		if _, err := os.Stat(filepath.Join(dir, "objects")); err != nil {
			if _, err := codehost.Run(dir, "git", "init", "--bare"); err != nil {
				os.RemoveAll(dir)
				return nil, err
			}
			// We could just say git fetch https://whatever later,
			// but this lets us say git fetch origin instead, which
			// is a little nicer. More importantly, using a named remote
			// avoids a problem with Git LFS. See golang.org/issue/25605.
			if _, err := codehost.Run(dir, "git", "remote", "add", "origin", r.remote); err != nil {
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

type repo struct {
	remote     string
	local      bool
	root       string
	dir        string
	canArchive bool

	refsOnce sync.Once
	refs     map[string]string
	refsErr  error
}

func (r *repo) Root() string {
	return r.root
}

// loadRefs loads heads and tags references from the remote into the map r.refs.
// Should only be called as r.refsOnce.Do(r.loadRefs).
func (r *repo) loadRefs() {
	// The git protocol sends all known refs and ls-remote filters them on the client side,
	// so we might as well record both heads and tags in one shot.
	// Most of the time we only care about tags but sometimes we care about heads too.
	out, err := codehost.Run(r.dir, "git", "ls-remote", "-q", r.remote)
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

func (r *repo) Tags(prefix string) ([]string, error) {
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

func (r *repo) Latest() (*codehost.RevInfo, error) {
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
func (r *repo) findRef(hash string) (ref string, ok bool) {
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

// statOrArchive tries to stat the given rev in the local repository,
// or else it tries to obtain an archive at the rev with the given arguments,
// or else it falls back to aggressive fetching and then a local stat.
// The archive step is an optimization for servers that support it
// (most do not, but maybe that will change), to let us minimize
// the amount of code downloaded.
func (r *repo) statOrArchive(rev string, archiveArgs ...string) (info *codehost.RevInfo, archive []byte, err error) {
	// Do we have this rev?
	r.refsOnce.Do(r.loadRefs)
	var hash string
	if k := "refs/tags/" + rev; r.refs[k] != "" {
		hash = r.refs[k]
	} else if k := "refs/heads/" + rev; r.refs[k] != "" {
		hash = r.refs[k]
		rev = hash
	} else if rev == "HEAD" && r.refs["HEAD"] != "" {
		hash = r.refs["HEAD"]
		rev = hash
	} else if len(rev) >= 5 && len(rev) <= 40 && codehost.AllHex(rev) {
		hash = rev
	} else {
		return nil, nil, fmt.Errorf("unknown revision %q", rev)
	}

	out, err := codehost.Run(r.dir, "git", "log", "-n1", "--format=format:%H", hash)
	if err == nil {
		hash = strings.TrimSpace(string(out))
		goto Found
	}

	// We don't have the rev. Can we fetch it?
	if r.local {
		return nil, nil, fmt.Errorf("unknown revision %q", rev)
	}

	if r.canArchive {
		// git archive with --remote requires a ref, not a hash.
		// Proceed only if we know a ref for this hash.
		if ref, ok := r.findRef(hash); ok {
			out, err := codehost.Run(r.dir, "git", "archive", "--format=zip", "--remote="+r.remote, "--prefix=prefix/", ref, archiveArgs)
			if err == nil {
				return &codehost.RevInfo{Version: rev}, out, nil
			}
			if bytes.Contains(err.(*codehost.RunError).Stderr, []byte("did not match any files")) {
				return nil, nil, fmt.Errorf("file not found")
			}
			if bytes.Contains(err.(*codehost.RunError).Stderr, []byte("Operation not supported by protocol")) {
				r.canArchive = false
			}
		}
	}

	// Maybe it's a prefix of a ref we know.
	// Iterating through all the refs is faster than doing unnecessary fetches.
	// This is not strictly correct, in that the short ref might be ambiguous
	// in the git repo as a whole, but not ambiguous in the list of named refs,
	// so that we will resolve it where the git server would not.
	// But this check avoids great expense, and preferring a known ref does
	// not seem like such a bad failure mode.
	if len(hash) >= 5 && len(hash) < 40 {
		var full string
		for _, h := range r.refs {
			if strings.HasPrefix(h, hash) {
				if full != "" {
					// Prefix is ambiguous even in the ref list!
					full = ""
					break
				}
				full = h
			}
		}
		if full != "" {
			hash = full
		}
	}

	// Fetch it.
	if len(hash) == 40 {
		name := hash
		if ref, ok := r.findRef(hash); ok {
			name = ref
		}
		if _, err = codehost.Run(r.dir, "git", "fetch", "--depth=1", r.remote, name); err == nil {
			goto Found
		}
		if !strings.Contains(err.Error(), "unadvertised object") && !strings.Contains(err.Error(), "no such remote ref") && !strings.Contains(err.Error(), "does not support shallow") {
			return nil, nil, err
		}
	}

	// It's a prefix, and we don't have a way to make the server resolve the prefix for us,
	// or it's a full hash but also an unadvertised object.
	// Download progressively more of the repo to look for it.

	// Fetch the main branch (non-shallow).
	if _, err := codehost.Run(r.dir, "git", "fetch", unshallow(r.dir), r.remote); err != nil {
		return nil, nil, err
	}
	if out, err := codehost.Run(r.dir, "git", "log", "-n1", "--format=format:%H", hash); err == nil {
		hash = strings.TrimSpace(string(out))
		goto Found
	}

	// Fetch all tags (non-shallow).
	if _, err := codehost.Run(r.dir, "git", "fetch", unshallow(r.dir), "-f", "--tags", r.remote); err != nil {
		return nil, nil, err
	}
	if out, err := codehost.Run(r.dir, "git", "log", "-n1", "--format=format:%H", hash); err == nil {
		hash = strings.TrimSpace(string(out))
		goto Found
	}

	// Fetch all branches (non-shallow).
	if _, err := codehost.Run(r.dir, "git", "fetch", unshallow(r.dir), "-f", r.remote, "refs/heads/*:refs/heads/*"); err != nil {
		return nil, nil, err
	}
	if out, err := codehost.Run(r.dir, "git", "log", "-n1", "--format=format:%H", hash); err == nil {
		hash = strings.TrimSpace(string(out))
		goto Found
	}

	// Fetch all refs (non-shallow).
	if _, err := codehost.Run(r.dir, "git", "fetch", unshallow(r.dir), "-f", r.remote, "refs/*:refs/*"); err != nil {
		return nil, nil, err
	}
	if out, err := codehost.Run(r.dir, "git", "log", "-n1", "--format=format:%H", hash); err == nil {
		hash = strings.TrimSpace(string(out))
		goto Found
	}
	return nil, nil, fmt.Errorf("cannot find hash %s", hash)
Found:

	if strings.HasPrefix(hash, rev) {
		rev = hash
	}

	out, err = codehost.Run(r.dir, "git", "log", "-n1", "--format=format:%ct", hash)
	if err != nil {
		return nil, nil, err
	}
	t, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid time from git log: %q", out)
	}

	info = &codehost.RevInfo{
		Name:    hash,
		Short:   codehost.ShortenSHA1(hash),
		Time:    time.Unix(t, 0).UTC(),
		Version: rev,
	}
	return info, nil, nil
}

func (r *repo) Stat(rev string) (*codehost.RevInfo, error) {
	// If the server will give us a git archive, we can pull the
	// commit ID and the commit time out of the archive.
	// We want an archive as small as possible (for speed),
	// but we have to specify a pattern that matches at least one file name.
	// The pattern here matches README, .gitignore, .gitattributes,
	// and go.mod (and some other incidental file names);
	// hopefully most repos will have at least one of these.
	info, archive, err := r.statOrArchive(rev, "[Rg.][Ego][A.i][Dmt][Miao][Edgt]*")
	if err != nil {
		return nil, err
	}
	if archive != nil {
		return zip2info(archive, info.Version)
	}
	return info, nil
}

func (r *repo) ReadFile(rev, file string, maxSize int64) ([]byte, error) {
	info, archive, err := r.statOrArchive(rev, file)
	if err != nil {
		return nil, err
	}
	if archive != nil {
		return zip2file(archive, file, maxSize)
	}
	out, err := codehost.Run(r.dir, "git", "cat-file", "blob", info.Name+":"+file)
	if err != nil {
		return nil, fmt.Errorf("file not found")
	}
	return out, nil
}

func (r *repo) ReadZip(rev, subdir string, maxSize int64) (zip io.ReadCloser, actualSubdir string, err error) {
	// TODO: Use maxSize or drop it.
	args := []string{}
	if subdir != "" {
		args = append(args, "--", subdir)
	}
	info, archive, err := r.statOrArchive(rev, args...)
	if err != nil {
		return nil, "", err
	}
	if archive == nil {
		archive, err = codehost.Run(r.dir, "git", "archive", "--format=zip", "--prefix=prefix/", info.Name, args)
		if err != nil {
			if bytes.Contains(err.(*codehost.RunError).Stderr, []byte("did not match any files")) {
				return nil, "", fmt.Errorf("file not found")
			}
			return nil, "", err
		}
	}

	return ioutil.NopCloser(bytes.NewReader(archive)), "", nil
}

func zip2info(archive []byte, rev string) (*codehost.RevInfo, error) {
	r, err := zip.NewReader(bytes.NewReader(archive), int64(len(archive)))
	if err != nil {
		return nil, err
	}
	if r.Comment == "" {
		return nil, fmt.Errorf("missing commit ID in git zip comment")
	}
	hash := r.Comment
	if len(hash) != 40 || !codehost.AllHex(hash) {
		return nil, fmt.Errorf("invalid commit ID in git zip comment")
	}
	if len(r.File) == 0 {
		return nil, fmt.Errorf("git zip has no files")
	}
	info := &codehost.RevInfo{
		Name:    hash,
		Short:   codehost.ShortenSHA1(hash),
		Time:    r.File[0].Modified.UTC(),
		Version: rev,
	}
	return info, nil
}

func zip2file(archive []byte, file string, maxSize int64) ([]byte, error) {
	r, err := zip.NewReader(bytes.NewReader(archive), int64(len(archive)))
	if err != nil {
		return nil, err
	}
	for _, f := range r.File {
		if f.Name != "prefix/"+file {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		defer rc.Close()
		l := &io.LimitedReader{R: rc, N: maxSize + 1}
		data, err := ioutil.ReadAll(l)
		if err != nil {
			return nil, err
		}
		if l.N <= 0 {
			return nil, fmt.Errorf("file %s too large", file)
		}
		return data, nil
	}
	return nil, fmt.Errorf("incomplete git zip archive: cannot find %s", file)
}
