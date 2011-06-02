// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"os"
	"strings"
)

// ErrNotFound is the error resulting if a path search failed to find an executable file.
var ErrNotFound = os.ErrorString("executable file not found in %PATH%")

func chkStat(file string) os.Error {
	d, err := os.Stat(file)
	if err != nil {
		return err
	}
	if d.IsRegular() {
		return nil
	}
	return os.EPERM
}

func findExecutable(file string, exts []string) (string, os.Error) {
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
	return ``, ErrNotFound
}

func LookPath(file string) (f string, err os.Error) {
	exts := []string{}
	if x := os.Getenv(`PATHEXT`); x != `` {
		exts = strings.Split(strings.ToLower(x), `;`, -1)
		for i, e := range exts {
			if e == `` || e[0] != '.' {
				exts[i] = `.` + e
			}
		}
	}
	if strings.Contains(file, `\`) || strings.Contains(file, `/`) {
		if f, err = findExecutable(file, exts); err == nil {
			return
		}
		return ``, &Error{file, err}
	}
	if pathenv := os.Getenv(`PATH`); pathenv == `` {
		if f, err = findExecutable(`.\`+file, exts); err == nil {
			return
		}
	} else {
		for _, dir := range strings.Split(pathenv, `;`, -1) {
			if f, err = findExecutable(dir+`\`+file, exts); err == nil {
				return
			}
		}
	}
	return ``, &Error{file, ErrNotFound}
}
