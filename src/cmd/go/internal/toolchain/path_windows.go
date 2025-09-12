// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package toolchain

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"cmd/go/internal/gover"
)

var pathExts = sync.OnceValue(func() []string {
	x := os.Getenv(`PATHEXT`)
	if x == "" {
		return []string{".com", ".exe", ".bat", ".cmd"}
	}

	var exts []string
	for e := range strings.SplitSeq(strings.ToLower(x), `;`) {
		if e == "" {
			continue
		}
		if e[0] != '.' {
			e = "." + e
		}
		exts = append(exts, e)
	}
	return exts
})

// pathDirs returns the directories in the system search path.
func pathDirs() []string {
	return filepath.SplitList(os.Getenv("PATH"))
}

// pathVersion returns the Go version implemented by the file
// described by de and info in directory dir.
// The analysis only uses the name itself; it does not run the program.
func pathVersion(dir string, de fs.DirEntry, info fs.FileInfo) (string, bool) {
	name, _, ok := cutExt(de.Name(), pathExts())
	if !ok {
		return "", false
	}
	v := gover.FromToolchain(name)
	if v == "" {
		return "", false
	}
	return v, true
}

// cutExt looks for any of the known extensions at the end of file.
// If one is found, cutExt returns the file name with the extension trimmed,
// the extension itself, and true to signal that an extension was found.
// Otherwise cutExt returns file, "", false.
func cutExt(file string, exts []string) (name, ext string, found bool) {
	i := strings.LastIndex(file, ".")
	if i < 0 {
		return file, "", false
	}
	for _, x := range exts {
		if strings.EqualFold(file[i:], x) {
			return file[:i], file[i:], true
		}
	}
	return file, "", false
}
