// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"fmt"
	"strconv"
	"strings"

	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
)

func ParseGopkgLock(file string, data []byte) (*modfile.File, error) {
	mf := new(modfile.File)
	var list []module.Version
	var r *module.Version
	for lineno, line := range strings.Split(string(data), "\n") {
		lineno++
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		line = strings.TrimSpace(line)
		if line == "[[projects]]" {
			list = append(list, module.Version{})
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
		mf.Require = append(mf.Require, &modfile.Require{Mod: r})
	}
	return mf, nil
}
