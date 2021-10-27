// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysis performs type and pointer analysis
// and generates mark-up for the Go source view.
//
// The Run method populates a Result object by running type and
// (optionally) pointer analysis.  The Result object is thread-safe
// and at all times may be accessed by a serving thread, even as it is
// progressively populated as analysis facts are derived.
//
// The Result is a mapping from each godoc file URL
// (e.g. /src/fmt/print.go) to information about that file.  The
// information is a list of HTML markup links and a JSON array of
// structured data values.  Some of the links call client-side
// JavaScript functions that index this array.
//
// The analysis computes mark-up for the following relations:
//
// IMPORTS: for each ast.ImportSpec, the package that it denotes.
//
// RESOLUTION: for each ast.Ident, its kind and type, and the location
// of its definition.
//
// METHOD SETS, IMPLEMENTS: for each ast.Ident defining a named type,
// its method-set, the set of interfaces it implements or is
// implemented by, and its size/align values.
//
// CALLERS, CALLEES: for each function declaration ('func' token), its
// callers, and for each call-site ('(' token), its callees.
//
// CALLGRAPH: the package docs include an interactive viewer for the
// intra-package call graph of "fmt".
//
// CHANNEL PEERS: for each channel operation make/<-/close, the set of
// other channel ops that alias the same channel(s).
//
// ERRORS: for each locus of a frontend (scanner/parser/type) error, the
// location is highlighted in red and hover text provides the compiler
// error message.
//
package analysis // import "golang.org/x/tools/godoc/analysis"

import (
	"io"
	"sort"
	"sync"
)

// -- links ------------------------------------------------------------

// A Link is an HTML decoration of the bytes [Start, End) of a file.
// Write is called before/after those bytes to emit the mark-up.
type Link interface {
	Start() int
	End() int
	Write(w io.Writer, _ int, start bool) // the godoc.LinkWriter signature
}

// -- fileInfo ---------------------------------------------------------

// FileInfo holds analysis information for the source file view.
// Clients must not mutate it.
type FileInfo struct {
	Data  []interface{} // JSON serializable values
	Links []Link        // HTML link markup
}

// A fileInfo is the server's store of hyperlinks and JSON data for a
// particular file.
type fileInfo struct {
	mu        sync.Mutex
	data      []interface{} // JSON objects
	links     []Link
	sorted    bool
	hasErrors bool // TODO(adonovan): surface this in the UI
}

// get returns the file info in external form.
// Callers must not mutate its fields.
func (fi *fileInfo) get() FileInfo {
	var r FileInfo
	// Copy slices, to avoid races.
	fi.mu.Lock()
	r.Data = append(r.Data, fi.data...)
	if !fi.sorted {
		sort.Sort(linksByStart(fi.links))
		fi.sorted = true
	}
	r.Links = append(r.Links, fi.links...)
	fi.mu.Unlock()
	return r
}

// PackageInfo holds analysis information for the package view.
// Clients must not mutate it.
type PackageInfo struct {
	CallGraph      []*PCGNodeJSON
	CallGraphIndex map[string]int
	Types          []*TypeInfoJSON
}

type pkgInfo struct {
	mu             sync.Mutex
	callGraph      []*PCGNodeJSON
	callGraphIndex map[string]int  // keys are (*ssa.Function).RelString()
	types          []*TypeInfoJSON // type info for exported types
}

// get returns the package info in external form.
// Callers must not mutate its fields.
func (pi *pkgInfo) get() PackageInfo {
	var r PackageInfo
	// Copy slices, to avoid races.
	pi.mu.Lock()
	r.CallGraph = append(r.CallGraph, pi.callGraph...)
	r.CallGraphIndex = pi.callGraphIndex
	r.Types = append(r.Types, pi.types...)
	pi.mu.Unlock()
	return r
}

// -- Result -----------------------------------------------------------

// Result contains the results of analysis.
// The result contains a mapping from filenames to a set of HTML links
// and JavaScript data referenced by the links.
type Result struct {
	mu        sync.Mutex           // guards maps (but not their contents)
	status    string               // global analysis status
	fileInfos map[string]*fileInfo // keys are godoc file URLs
	pkgInfos  map[string]*pkgInfo  // keys are import paths
}

// fileInfo returns the fileInfo for the specified godoc file URL,
// constructing it as needed.  Thread-safe.
func (res *Result) fileInfo(url string) *fileInfo {
	res.mu.Lock()
	fi, ok := res.fileInfos[url]
	if !ok {
		if res.fileInfos == nil {
			res.fileInfos = make(map[string]*fileInfo)
		}
		fi = new(fileInfo)
		res.fileInfos[url] = fi
	}
	res.mu.Unlock()
	return fi
}

// Status returns a human-readable description of the current analysis status.
func (res *Result) Status() string {
	res.mu.Lock()
	defer res.mu.Unlock()
	return res.status
}

// FileInfo returns new slices containing opaque JSON values and the
// HTML link markup for the specified godoc file URL.  Thread-safe.
// Callers must not mutate the elements.
// It returns "zero" if no data is available.
//
func (res *Result) FileInfo(url string) (fi FileInfo) {
	return res.fileInfo(url).get()
}

// pkgInfo returns the pkgInfo for the specified import path,
// constructing it as needed.  Thread-safe.
func (res *Result) pkgInfo(importPath string) *pkgInfo {
	res.mu.Lock()
	pi, ok := res.pkgInfos[importPath]
	if !ok {
		if res.pkgInfos == nil {
			res.pkgInfos = make(map[string]*pkgInfo)
		}
		pi = new(pkgInfo)
		res.pkgInfos[importPath] = pi
	}
	res.mu.Unlock()
	return pi
}

// PackageInfo returns new slices of JSON values for the callgraph and
// type info for the specified package.  Thread-safe.
// Callers must not mutate its fields.
// PackageInfo returns "zero" if no data is available.
//
func (res *Result) PackageInfo(importPath string) PackageInfo {
	return res.pkgInfo(importPath).get()
}

type linksByStart []Link

func (a linksByStart) Less(i, j int) bool { return a[i].Start() < a[j].Start() }
func (a linksByStart) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a linksByStart) Len() int           { return len(a) }
