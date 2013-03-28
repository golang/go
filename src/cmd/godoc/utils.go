// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains support functionality for godoc.

package main

import (
	pathpkg "path"
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
	if isText, found := textExt[pathpkg.Ext(filename)]; found {
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
