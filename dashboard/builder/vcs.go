// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/xml"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// Repo represents a mercurial repository.
type Repo struct {
	Path string
	sync.Mutex
}

// RemoteRepo constructs a *Repo representing a remote repository.
func RemoteRepo(url string) *Repo {
	return &Repo{
		Path: url,
	}
}

// Clone clones the current Repo to a new destination
// returning a new *Repo if successful.
func (r *Repo) Clone(path, rev string) (*Repo, error) {
	r.Lock()
	defer r.Unlock()
	if err := run(*cmdTimeout, nil, *buildroot, r.hgCmd("clone", "-r", rev, r.Path, path)...); err != nil {
		return nil, err
	}
	return &Repo{
		Path: path,
	}, nil
}

// UpdateTo updates the working copy of this Repo to the
// supplied revision.
func (r *Repo) UpdateTo(hash string) error {
	r.Lock()
	defer r.Unlock()
	return run(*cmdTimeout, nil, r.Path, r.hgCmd("update", hash)...)
}

// Exists reports whether this Repo represents a valid Mecurial repository.
func (r *Repo) Exists() bool {
	fi, err := os.Stat(filepath.Join(r.Path, ".hg"))
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
	return run(*cmdTimeout, nil, r.Path, r.hgCmd("pull")...)
}

// Log returns the changelog for this repository.
func (r *Repo) Log() ([]HgLog, error) {
	if err := r.Pull(); err != nil {
		return nil, err
	}
	const N = 50 // how many revisions to grab

	r.Lock()
	defer r.Unlock()
	data, _, err := runLog(*cmdTimeout, nil, r.Path, r.hgCmd("log",
		"--encoding=utf-8",
		"--limit="+strconv.Itoa(N),
		"--template="+xmlLogTemplate)...,
	)
	if err != nil {
		return nil, err
	}

	var logStruct struct {
		Log []HgLog
	}
	err = xml.Unmarshal([]byte("<Top>"+data+"</Top>"), &logStruct)
	if err != nil {
		log.Printf("unmarshal hg log: %v", err)
		return nil, err
	}
	return logStruct.Log, nil
}

// FullHash returns the full hash for the given Mercurial revision.
func (r *Repo) FullHash(rev string) (string, error) {
	r.Lock()
	defer r.Unlock()
	s, _, err := runLog(*cmdTimeout, nil, r.Path,
		r.hgCmd("log",
			"--encoding=utf-8",
			"--rev="+rev,
			"--limit=1",
			"--template={node}")...,
	)
	if err != nil {
		return "", nil
	}
	s = strings.TrimSpace(s)
	if s == "" {
		return "", fmt.Errorf("cannot find revision")
	}
	if len(s) != 40 {
		return "", fmt.Errorf("hg returned invalid hash " + s)
	}
	return s, nil
}

func (r *Repo) hgCmd(args ...string) []string {
	return append([]string{"hg", "--config", "extensions.codereview=!"}, args...)
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
