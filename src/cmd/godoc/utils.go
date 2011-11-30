// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains support functionality for godoc.

package main

import (
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// An RWValue wraps a value and permits mutually exclusive
// access to it and records the time the value was last set.
//
type RWValue struct {
	mutex     sync.RWMutex
	value     interface{}
	timestamp time.Time // time of last set()
}

func (v *RWValue) set(value interface{}) {
	v.mutex.Lock()
	v.value = value
	v.timestamp = time.Now()
	v.mutex.Unlock()
}

func (v *RWValue) get() (interface{}, time.Time) {
	v.mutex.RLock()
	defer v.mutex.RUnlock()
	return v.value, v.timestamp
}

// TODO(gri) For now, using os.Getwd() is ok here since the functionality
//           based on this code is not invoked for the appengine version,
//           but this is fragile. Determine what the right thing to do is,
//           here (possibly have some Getwd-equivalent in FileSystem).
var cwd, _ = os.Getwd() // ignore errors

// canonicalizePaths takes a list of (directory/file) paths and returns
// the list of corresponding absolute paths in sorted (increasing) order.
// Relative paths are assumed to be relative to the current directory,
// empty and duplicate paths as well as paths for which filter(path) is
// false are discarded. filter may be nil in which case it is not used.
//
func canonicalizePaths(list []string, filter func(path string) bool) []string {
	i := 0
	for _, path := range list {
		path = strings.TrimSpace(path)
		if len(path) == 0 {
			continue // ignore empty paths (don't assume ".")
		}
		// len(path) > 0: normalize path
		if filepath.IsAbs(path) {
			path = filepath.Clean(path)
		} else {
			path = filepath.Join(cwd, path)
		}
		// we have a non-empty absolute path
		if filter != nil && !filter(path) {
			continue
		}
		// keep the path
		list[i] = path
		i++
	}
	list = list[0:i]

	// sort the list and remove duplicate entries
	sort.Strings(list)
	i = 0
	prev := ""
	for _, path := range list {
		if path != prev {
			list[i] = path
			i++
			prev = path
		}
	}

	return list[0:i]
}

// writeFileAtomically writes data to a temporary file and then
// atomically renames that file to the file named by filename.
//
func writeFileAtomically(filename string, data []byte) error {
	// TODO(gri) this won't work on appengine
	f, err := ioutil.TempFile(filepath.Split(filename))
	if err != nil {
		return err
	}
	n, err := f.Write(data)
	f.Close()
	if err != nil {
		return err
	}
	if n < len(data) {
		return io.ErrShortWrite
	}
	return os.Rename(f.Name(), filename)
}

// isText returns true if a significant prefix of s looks like correct UTF-8;
// that is, if it is likely that s is human-readable text.
//
func isText(s []byte) bool {
	const max = 1024 // at least utf8.UTFMax
	if len(s) > max {
		s = s[0:max]
	}
	for i, c := range string(s) {
		if i+utf8.UTFMax > len(s) {
			// last char may be incomplete - ignore
			break
		}
		if c == 0xFFFD || c < ' ' && c != '\n' && c != '\t' && c != '\f' {
			// decoding error or control character - not a text file
			return false
		}
	}
	return true
}

// TODO(gri): Should have a mapping from extension to handler, eventually.

// textExt[x] is true if the extension x indicates a text file, and false otherwise.
var textExt = map[string]bool{
	".css": false, // must be served raw
	".js":  false, // must be served raw
}

// isTextFile returns true if the file has a known extension indicating
// a text file, or if a significant chunk of the specified file looks like
// correct UTF-8; that is, if it is likely that the file contains human-
// readable text.
//
func isTextFile(filename string) bool {
	// if the extension is known, use it for decision making
	if isText, found := textExt[filepath.Ext(filename)]; found {
		return isText
	}

	// the extension is not known; read an initial chunk
	// of the file and check if it looks like text
	f, err := fs.Open(filename)
	if err != nil {
		return false
	}
	defer f.Close()

	var buf [1024]byte
	n, err := f.Read(buf[0:])
	if err != nil {
		return false
	}

	return isText(buf[0:n])
}
