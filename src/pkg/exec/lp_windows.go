// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"os"
	"strings"
)

func chkStat(file string) bool {
	d, err := os.Stat(file)
	if err != nil {
		return false
	}
	return d.IsRegular()
}

func canExec(file string, exts []string) (string, bool) {
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
		if f := file + e; chkStat(f) {
			return f, true
		}
	}
	return ``, false
}

func LookPath(file string) (string, os.Error) {
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
		if f, ok := canExec(file, exts); ok {
			return f, nil
		}
		return ``, &os.PathError{"lookpath", file, os.ENOENT}
	}
	if pathenv := os.Getenv(`PATH`); pathenv == `` {
		if f, ok := canExec(`.\`+file, exts); ok {
			return f, nil
		}
	} else {
		for _, dir := range strings.Split(pathenv, `;`, -1) {
			if f, ok := canExec(dir+`\`+file, exts); ok {
				return f, nil
			}
		}
	}
	return ``, &os.PathError{"lookpath", file, os.ENOENT}
}
