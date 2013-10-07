// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/xml"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"code.google.com/p/go.tools/go/vcs"
)

// Repo represents a mercurial repository.
type Repo struct {
	Path   string
	Master *vcs.RepoRoot
	sync.Mutex
}

// RemoteRepo constructs a *Repo representing a remote repository.
func RemoteRepo(url, path string) (*Repo, error) {
	rr, err := vcs.RepoRootForImportPath(url, *verbose)
	if err != nil {
		return nil, err
	}
	return &Repo{
		Path:   path,
		Master: rr,
	}, nil
}

// Clone clones the current Repo to a new destination
// returning a new *Repo if successful.
func (r *Repo) Clone(path, rev string) (*Repo, error) {
	r.Lock()
	defer r.Unlock()

	err := timeout(*cmdTimeout, func() error {
		downloadPath := r.Path
		if !r.Exists() {
			downloadPath = r.Master.Repo
		}

		err := r.Master.VCS.CreateAtRev(path, downloadPath, rev)
		if err != nil {
			return err
		}
		return r.Master.VCS.TagSync(path, "")
	})
	if err != nil {
		return nil, err
	}
	return &Repo{
		Path:   path,
		Master: r.Master,
	}, nil
}

// UpdateTo updates the working copy of this Repo to the
// supplied revision.
func (r *Repo) UpdateTo(hash string) error {
	r.Lock()
	defer r.Unlock()

	return timeout(*cmdTimeout, func() error {
		return r.Master.VCS.TagSync(r.Path, hash)
	})
}

// Exists reports whether this Repo represents a valid Mecurial repository.
func (r *Repo) Exists() bool {
	fi, err := os.Stat(filepath.Join(r.Path, "."+r.Master.VCS.Cmd))
	if err != nil {
		return false
	}
	return fi.IsDir()
}

// Pull pulls changes from the default path, that is, the path
// this Repo was cloned from.
func (r *Repo) Pull() error {
	r.Lock()
	defer r.Unlock()

	return timeout(*cmdTimeout, func() error {
		return r.Master.VCS.Download(r.Path)
	})
}

// Log returns the changelog for this repository.
func (r *Repo) Log() ([]HgLog, error) {
	if err := r.Pull(); err != nil {
		return nil, err
	}
	r.Lock()
	defer r.Unlock()

	var logStruct struct {
		Log []HgLog
	}
	err := timeout(*cmdTimeout, func() error {
		data, err := r.Master.VCS.Log(r.Path, xmlLogTemplate)
		if err != nil {
			return err
		}

		err = xml.Unmarshal([]byte("<Top>"+string(data)+"</Top>"), &logStruct)
		if err != nil {
			return fmt.Errorf("unmarshal %s log: %v", r.Master.VCS, err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return logStruct.Log, nil
}

// FullHash returns the full hash for the given Mercurial revision.
func (r *Repo) FullHash(rev string) (string, error) {
	r.Lock()
	defer r.Unlock()

	var hash string
	err := timeout(*cmdTimeout, func() error {
		data, err := r.Master.VCS.LogAtRev(r.Path, rev, "{node}")
		if err != nil {
			return err
		}

		s := strings.TrimSpace(string(data))
		if s == "" {
			return fmt.Errorf("cannot find revision")
		}
		if len(s) != 40 {
			return fmt.Errorf("%s returned invalid hash: %s", r.Master.VCS, s)
		}
		hash = s
		return nil
	})
	if err != nil {
		return "", err
	}
	return hash, nil
}

// HgLog represents a single Mercurial revision.
type HgLog struct {
	Hash   string
	Author string
	Date   string
	Desc   string
	Parent string

	// Internal metadata
	added bool
}

// xmlLogTemplate is a template to pass to Mercurial to make
// hg log print the log in valid XML for parsing with xml.Unmarshal.
const xmlLogTemplate = `
        <Log>
        <Hash>{node|escape}</Hash>
        <Parent>{parent|escape}</Parent>
        <Author>{author|escape}</Author>
        <Date>{date|rfc3339date}</Date>
        <Desc>{desc|escape}</Desc>
        </Log>
`
