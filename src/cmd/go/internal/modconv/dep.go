// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"fmt"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"

	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
)

func ParseGopkgLock(file string, data []byte) (*modfile.File, error) {
	type pkg struct {
		Path    string
		Version string
		Source  string
	}
	mf := new(modfile.File)
	var list []pkg
	var r *pkg
	for lineno, line := range strings.Split(string(data), "\n") {
		lineno++
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		line = strings.TrimSpace(line)
		if line == "[[projects]]" {
			list = append(list, pkg{})
			r = &list[len(list)-1]
			continue
		}
		if strings.HasPrefix(line, "[") {
			r = nil
			continue
		}
		if r == nil {
			continue
		}
		i := strings.Index(line, "=")
		if i < 0 {
			continue
		}
		key := strings.TrimSpace(line[:i])
		val := strings.TrimSpace(line[i+1:])
		if len(val) >= 2 && val[0] == '"' && val[len(val)-1] == '"' {
			q, err := strconv.Unquote(val) // Go unquoting, but close enough for now
			if err != nil {
				return nil, fmt.Errorf("%s:%d: invalid quoted string: %v", file, lineno, err)
			}
			val = q
		}
		switch key {
		case "name":
			r.Path = val
		case "source":
			r.Source = val
		case "revision", "version":
			// Note: key "version" should take priority over "revision",
			// and it does, because dep writes toml keys in alphabetical order,
			// so we see version (if present) second.
			if key == "version" {
				if !semver.IsValid(val) || semver.Canonical(val) != val {
					break
				}
			}
			r.Version = val
		}
	}
	for _, r := range list {
		if r.Path == "" || r.Version == "" {
			return nil, fmt.Errorf("%s: empty [[projects]] stanza (%s)", file, r.Path)
		}
		mf.Require = append(mf.Require, &modfile.Require{Mod: module.Version{Path: r.Path, Version: r.Version}})

		if r.Source != "" {
			// Convert "source" to import path, such as
			// git@test.com:x/y.git and https://test.com/x/y.git.
			// We get "test.com/x/y" at last.
			source, err := decodeSource(r.Source)
			if err != nil {
				return nil, err
			}
			old := module.Version{Path: r.Path, Version: r.Version}
			new := module.Version{Path: source, Version: r.Version}
			mf.Replace = append(mf.Replace, &modfile.Replace{Old: old, New: new})
		}
	}
	return mf, nil
}

var scpSyntaxReg = regexp.MustCompile(`^([a-zA-Z0-9_]+)@([a-zA-Z0-9._-]+):(.*)$`)

func decodeSource(source string) (string, error) {
	var u *url.URL
	var p string
	if m := scpSyntaxReg.FindStringSubmatch(source); m != nil {
		// Match SCP-like syntax and convert it to a URL.
		// Eg, "git@github.com:user/repo" becomes
		// "ssh://git@github.com/user/repo".
		u = &url.URL{
			Scheme: "ssh",
			User:   url.User(m[1]),
			Host:   m[2],
			Path:   "/" + m[3],
		}
	} else {
		var err error
		u, err = url.Parse(source)
		if err != nil {
			return "", fmt.Errorf("%q is not a valid URI", source)
		}
	}

	// If no scheme was passed, then the entire path will have been put into
	// u.Path. Either way, construct the normalized path correctly.
	if u.Host == "" {
		p = source
	} else {
		p = path.Join(u.Host, u.Path)
	}
	p = strings.TrimSuffix(p, ".git")
	p = strings.TrimSuffix(p, ".hg")
	return p, nil
}
