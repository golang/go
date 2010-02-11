// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Mapping data structure.

package main

import (
	"fmt"
	"io"
	"os"
	pathutil "path"
	"strings"
)


// A Mapping object maps relative paths (e.g. from URLs)
// to absolute paths (of the file system) and vice versa.
//
// A Mapping object consists of a list of individual mappings
// of the form: prefix -> path which are interpreted as follows:
// A relative path of the form prefix/tail is to be mapped to
// the absolute path/tail, if that absolute path exists in the file
// system. Given a Mapping object, a relative path is mapped to an
// absolute path by trying each of the individual mappings in order,
// until a valid mapping is found. For instance, for the mapping:
//
//	user   -> /home/user
//      public -> /home/user/public
//	public -> /home/build/public
//
// the relative paths below are mapped to absolute paths as follows:
//
//	user/foo                -> /home/user/foo
//      public/net/rpc/file1.go -> /home/user/public/net/rpc/file1.go
//
// If there is no /home/user/public/net/rpc/file2.go, the next public
// mapping entry is used to map the relative path to:
//
//	public/net/rpc/file2.go -> /home/build/public/net/rpc/file2.go
//
// (assuming that file exists).
//
type Mapping struct {
	list []mapping
}


type mapping struct {
	prefix, path string
}


// Init initializes the Mapping from a list of ':'-separated
// paths. Empty paths are ignored; relative paths are assumed
// to be relative to the current working directory and converted
// to absolute paths. For each path of the form:
//
//	dirname/localname
//
// a mapping
//
//	localname -> path
//
// is added to the Mapping object, in the order of occurrence.
// For instance, the argument:
//
//	/home/user:/home/build/public
//
// leads to the following mapping:
//
//	user   -> /home/user
//      public -> /home/build/public
//
func (m *Mapping) Init(paths string) {
	cwd, _ := os.Getwd() // ignore errors

	pathlist := strings.Split(paths, ":", 0)

	list := make([]mapping, len(pathlist))
	n := 0 // number of mappings

	for _, path := range pathlist {
		if len(path) == 0 {
			// ignore empty paths (don't assume ".")
			continue
		}

		// len(path) > 0: normalize path
		if path[0] != '/' {
			path = pathutil.Join(cwd, path)
		} else {
			path = pathutil.Clean(path)
		}

		// check if mapping exists already
		var i int
		for i = 0; i < n; i++ {
			if path == list[i].path {
				break
			}
		}

		// add mapping if it is new
		if i >= n {
			_, prefix := pathutil.Split(path)
			list[i] = mapping{prefix, path}
			n++
		}
	}

	m.list = list[0:n]
}


// IsEmpty returns true if there are no mappings specified.
func (m *Mapping) IsEmpty() bool { return len(m.list) == 0 }


// Fprint prints the mapping.
func (m *Mapping) Fprint(w io.Writer) {
	for _, e := range m.list {
		fmt.Fprintf(w, "\t%s -> %s\n", e.prefix, e.path)
	}
}


func split(path string) (head, tail string) {
	i := strings.Index(path, "/")
	if i > 0 {
		// 0 < i < len(path)
		return path[0:i], path[i+1:]
	}
	return "", path
}


// ToAbsolute maps a relative path to an absolute path using the Mapping
// specified by the receiver. If the path cannot be mapped, the empty
// string is returned.
//
func (m *Mapping) ToAbsolute(path string) string {
	for _, e := range m.list {
		if strings.HasPrefix(path, e.path) {
			// /absolute/prefix/foo -> prefix/foo
			return pathutil.Join(e.prefix, path[len(e.path):]) // Join will remove a trailing '/'
		}
	}
	return "" // no match
}


// ToRelative maps an absolute path to a relative path using the Mapping
// specified by the receiver. If the path cannot be mapped, the empty
// string is returned.
//
func (m *Mapping) ToRelative(path string) string {
	prefix, tail := split(path)
	for _, e := range m.list {
		switch {
		case e.prefix == prefix:
			// use tail
		case e.prefix == "":
			tail = path
		default:
			continue // no match
		}
		abspath := pathutil.Join(e.path, tail)
		if _, err := os.Stat(abspath); err == nil {
			return abspath
		}
	}

	return "" // no match
}
