// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"errors"
	pathpkg "path"
	"sort"
	"strings"
	"time"

	"cmd/go/internal/modfetch/bitbucket"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modfetch/github"
	"cmd/go/internal/modfetch/googlesource"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
)

// A Repo represents a repository storing all versions of a single module.
type Repo interface {
	// ModulePath returns the module path.
	ModulePath() string

	// Versions lists all known versions with the given prefix.
	// Pseudo-versions are not included.
	// Versions should be returned sorted in semver order
	// (implementations can use SortVersions).
	Versions(prefix string) (tags []string, err error)

	// Stat returns information about the revision rev.
	// A revision can be any identifier known to the underlying service:
	// commit hash, branch, tag, and so on.
	Stat(rev string) (*RevInfo, error)

	// Latest returns the latest revision on the default branch,
	// whatever that means in the underlying source code repository.
	// It is only used when there are no tagged versions.
	Latest() (*RevInfo, error)

	// GoMod returns the go.mod file for the given version.
	GoMod(version string) (data []byte, err error)

	// Zip downloads a zip file for the given version
	// to a new file in a given temporary directory.
	// It returns the name of the new file.
	// The caller should remove the file when finished with it.
	Zip(version, tmpdir string) (tmpfile string, err error)
}

// A Rev describes a single revision in a module repository.
type RevInfo struct {
	Version string    // version string
	Name    string    // complete ID in underlying repository
	Short   string    // shortened ID, for use in pseudo-version
	Time    time.Time // commit time
}

// Lookup returns the module with the given module path.
func Lookup(path string) (Repo, error) {
	if proxyURL != "" {
		return lookupProxy(path)
	}
	if code, err := lookupCodeHost(path, false); err != errNotHosted {
		if err != nil {
			return nil, err
		}
		return newCodeRepo(code, path)
	}
	return lookupCustomDomain(path)
}

func Import(path string, allowed func(module.Version) bool) (Repo, *RevInfo, error) {
	try := func(path string) (Repo, *RevInfo, error) {
		r, err := Lookup(path)
		if err != nil {
			return nil, nil, err
		}
		info, err := Query(path, "latest", allowed)
		if err != nil {
			return nil, nil, err
		}
		_, err = r.GoMod(info.Version)
		if err != nil {
			return nil, nil, err
		}
		return r, info, nil
	}

	var firstErr error
	for {
		r, info, err := try(path)
		if err == nil {
			return r, info, nil
		}
		if firstErr == nil {
			firstErr = err
		}
		p := pathpkg.Dir(path)
		if p == "." {
			break
		}
		path = p
	}
	return nil, nil, firstErr
}

var errNotHosted = errors.New("not hosted")

var isTest bool

func lookupCodeHost(path string, customDomain bool) (codehost.Repo, error) {
	switch {
	case strings.HasPrefix(path, "github.com/"):
		return github.Lookup(path)
	case strings.HasPrefix(path, "bitbucket.org/"):
		return bitbucket.Lookup(path)
	case customDomain && strings.HasSuffix(path[:strings.Index(path, "/")+1], ".googlesource.com/") ||
		isTest && strings.HasPrefix(path, "go.googlesource.com/scratch"):
		return googlesource.Lookup(path)
	case strings.HasPrefix(path, "gopkg.in/"):
		return gopkginLookup(path)
	}
	return nil, errNotHosted
}

func SortVersions(list []string) {
	sort.Slice(list, func(i, j int) bool {
		cmp := semver.Compare(list[i], list[j])
		if cmp != 0 {
			return cmp < 0
		}
		return list[i] < list[j]
	})
}
