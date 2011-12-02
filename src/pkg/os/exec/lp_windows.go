// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"errors"
	"os"
	"strings"
)

// ErrNotFound is the error resulting if a path search failed to find an executable file.
var ErrNotFound = errors.New("executable file not found in %PATH%")

func chkStat(file string) error {
	d, err := os.Stat(file)
	if err != nil {
		return err
	}
	if d.IsDir() {
		return os.EPERM
	}
	return nil
}

func findExecutable(file string, exts []string) (string, error) {
	if len(exts) == 0 {
		return file, chkStat(file)
	}
	f := strings.ToLower(file)
	for _, e := range exts {
		if strings.HasSuffix(f, e) {
			return file, chkStat(file)
		}
	}
	for _, e := range exts {
		if f := file + e; chkStat(f) == nil {
			return f, nil
		}
	}
	return ``, os.ENOENT
}

func LookPath(file string) (f string, err error) {
	x := os.Getenv(`PATHEXT`)
	if x == `` {
		x = `.COM;.EXE;.BAT;.CMD`
	}
	exts := []string{}
	for _, e := range strings.Split(strings.ToLower(x), `;`) {
		if e == "" {
			continue
		}
		if e[0] != '.' {
			e = "." + e
		}
		exts = append(exts, e)
	}
	if strings.IndexAny(file, `:\/`) != -1 {
		if f, err = findExecutable(file, exts); err == nil {
			return
		}
		return ``, &Error{file, err}
	}
	if f, err = findExecutable(`.\`+file, exts); err == nil {
		return
	}
	if pathenv := os.Getenv(`PATH`); pathenv != `` {
		for _, dir := range strings.Split(pathenv, `;`) {
			if f, err = findExecutable(dir+`\`+file, exts); err == nil {
				return
			}
		}
	}
	return ``, &Error{file, ErrNotFound}
}
