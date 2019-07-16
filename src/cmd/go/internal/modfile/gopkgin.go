// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: Figure out what gopkg.in should do.

package modfile

import "strings"

// ParseGopkgIn splits gopkg.in import paths into their constituent parts
func ParseGopkgIn(path string) (root, repo, major, subdir string, ok bool) {
	if !strings.HasPrefix(path, "gopkg.in/") {
		return
	}
	f := strings.Split(path, "/")
	if len(f) >= 2 {
		if elem, v, ok := dotV(f[1]); ok {
			root = strings.Join(f[:2], "/")
			repo = "github.com/go-" + elem + "/" + elem
			major = v
			subdir = strings.Join(f[2:], "/")
			return root, repo, major, subdir, true
		}
	}
	if len(f) >= 3 {
		if elem, v, ok := dotV(f[2]); ok {
			root = strings.Join(f[:3], "/")
			repo = "github.com/" + f[1] + "/" + elem
			major = v
			subdir = strings.Join(f[3:], "/")
			return root, repo, major, subdir, true
		}
	}
	return
}

func dotV(name string) (elem, v string, ok bool) {
	i := len(name) - 1
	for i >= 0 && '0' <= name[i] && name[i] <= '9' {
		i--
	}
	if i <= 2 || i+1 >= len(name) || name[i-1] != '.' || name[i] != 'v' || name[i+1] == '0' && len(name) != i+2 {
		return "", "", false
	}
	return name[:i-1], name[i:], true
}
