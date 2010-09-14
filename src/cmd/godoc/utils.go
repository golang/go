// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains support functionality for godoc.

package main

import (
	"io"
	"io/ioutil"
	"os"
	pathutil "path"
	"sort"
	"strings"
	"sync"
	"time"
)


// An RWValue wraps a value and permits mutually exclusive
// access to it and records the time the value was last set.
type RWValue struct {
	mutex     sync.RWMutex
	value     interface{}
	timestamp int64 // time of last set(), in seconds since epoch
}


func (v *RWValue) set(value interface{}) {
	v.mutex.Lock()
	v.value = value
	v.timestamp = time.Seconds()
	v.mutex.Unlock()
}


func (v *RWValue) get() (interface{}, int64) {
	v.mutex.RLock()
	defer v.mutex.RUnlock()
	return v.value, v.timestamp
}


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
		if path[0] != '/' {
			path = pathutil.Join(cwd, path)
		} else {
			path = pathutil.Clean(path)
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
	sort.SortStrings(list)
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
func writeFileAtomically(filename string, data []byte) os.Error {
	f, err := ioutil.TempFile(cwd, filename)
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
