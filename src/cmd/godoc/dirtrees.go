// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code dealing with package directory trees.

package main

import (
	"bytes"
	"go/doc"
	"go/parser"
	"io/ioutil"
	"os"
	pathutil "path"
	"strings"
	"unicode"
)


type Directory struct {
	Depth int
	Path  string // includes Name
	Name  string
	Text  string       // package documentation, if any
	Dirs  []*Directory // subdirectories
}


func isGoFile(f *os.FileInfo) bool {
	return f.IsRegular() &&
		!strings.HasPrefix(f.Name, ".") && // ignore .files
		pathutil.Ext(f.Name) == ".go"
}


func isPkgFile(f *os.FileInfo) bool {
	return isGoFile(f) &&
		!strings.HasSuffix(f.Name, "_test.go") // ignore test files
}


func isPkgDir(f *os.FileInfo) bool {
	return f.IsDirectory() && len(f.Name) > 0 && f.Name[0] != '_'
}


func firstSentence(s string) string {
	i := -1 // index+1 of first terminator (punctuation ending a sentence)
	j := -1 // index+1 of first terminator followed by white space
	prev := 'A'
	for k, ch := range s {
		k1 := k + 1
		if ch == '.' || ch == '!' || ch == '?' {
			if i < 0 {
				i = k1 // first terminator
			}
			if k1 < len(s) && s[k1] <= ' ' {
				if j < 0 {
					j = k1 // first terminator followed by white space
				}
				if !unicode.IsUpper(prev) {
					j = k1
					break
				}
			}
		}
		prev = ch
	}

	if j < 0 {
		// use the next best terminator
		j = i
		if j < 0 {
			// no terminator at all, use the entire string
			j = len(s)
		}
	}

	return s[0:j]
}


type treeBuilder struct {
	pathFilter func(string) bool
	maxDepth   int
}


func (b *treeBuilder) newDirTree(path, name string, depth int) *Directory {
	if b.pathFilter != nil && !b.pathFilter(path) {
		return nil
	}

	if depth >= b.maxDepth {
		// return a dummy directory so that the parent directory
		// doesn't get discarded just because we reached the max
		// directory depth
		return &Directory{depth, path, name, "", nil}
	}

	list, _ := ioutil.ReadDir(path) // ignore errors

	// determine number of subdirectories and package files
	ndirs := 0
	nfiles := 0
	var synopses [4]string // prioritized package documentation (0 == highest priority)
	for _, d := range list {
		switch {
		case isPkgDir(d):
			ndirs++
		case isPkgFile(d):
			nfiles++
			if synopses[0] == "" {
				// no "optimal" package synopsis yet; continue to collect synopses
				file, err := parser.ParseFile(pathutil.Join(path, d.Name), nil,
					parser.ParseComments|parser.PackageClauseOnly)
				if err == nil && file.Doc != nil {
					// prioritize documentation
					i := -1
					switch file.Name.Name {
					case name:
						i = 0 // normal case: directory name matches package name
					case fakePkgName:
						i = 1 // synopses for commands
					case "main":
						i = 2 // directory contains a main package
					default:
						i = 3 // none of the above
					}
					if 0 <= i && i < len(synopses) && synopses[i] == "" {
						synopses[i] = firstSentence(doc.CommentText(file.Doc))
					}
				}
			}
		}
	}

	// create subdirectory tree
	var dirs []*Directory
	if ndirs > 0 {
		dirs = make([]*Directory, ndirs)
		i := 0
		for _, d := range list {
			if isPkgDir(d) {
				dd := b.newDirTree(pathutil.Join(path, d.Name), d.Name, depth+1)
				if dd != nil {
					dirs[i] = dd
					i++
				}
			}
		}
		dirs = dirs[0:i]
	}

	// if there are no package files and no subdirectories
	// (with package files), ignore the directory
	if nfiles == 0 && len(dirs) == 0 {
		return nil
	}

	// select the highest-priority synopsis for the directory entry, if any
	synopsis := ""
	for _, synopsis = range synopses {
		if synopsis != "" {
			break
		}
	}

	return &Directory{depth, path, name, synopsis, dirs}
}


// Maximum directory depth, adjust as needed.
const maxDirDepth = 24

// newDirectory creates a new package directory tree with at most maxDepth
// levels, anchored at root. The result tree is pruned such that it only
// contains directories that contain package files or that contain
// subdirectories containing package files (transitively). If a non-nil
// pathFilter is provided, directory paths additionally must be accepted
// by the filter (i.e., pathFilter(path) must be true). If maxDepth is
// too shallow, the leaf nodes are assumed to contain package files even if
// their contents are not known (i.e., in this case the tree may contain
// directories w/o any package files).
//
func newDirectory(root string, pathFilter func(string) bool, maxDepth int) *Directory {
	d, err := os.Lstat(root)
	if err != nil || !isPkgDir(d) {
		return nil
	}
	b := treeBuilder{pathFilter, maxDepth}
	return b.newDirTree(root, d.Name, 0)
}


func (dir *Directory) writeLeafs(buf *bytes.Buffer) {
	if dir != nil {
		if len(dir.Dirs) == 0 {
			buf.WriteString(dir.Path)
			buf.WriteByte('\n')
			return
		}

		for _, d := range dir.Dirs {
			d.writeLeafs(buf)
		}
	}
}


func (dir *Directory) walk(c chan<- *Directory, skipRoot bool) {
	if dir != nil {
		if !skipRoot {
			c <- dir
		}
		for _, d := range dir.Dirs {
			d.walk(c, false)
		}
	}
}


func (dir *Directory) iter(skipRoot bool) <-chan *Directory {
	c := make(chan *Directory)
	go func() {
		dir.walk(c, skipRoot)
		close(c)
	}()
	return c
}


func (dir *Directory) lookupLocal(name string) *Directory {
	for _, d := range dir.Dirs {
		if d.Name == name {
			return d
		}
	}
	return nil
}


// lookup looks for the *Directory for a given path, relative to dir.
func (dir *Directory) lookup(path string) *Directory {
	d := strings.Split(dir.Path, "/", -1)
	p := strings.Split(path, "/", -1)
	i := 0
	for i < len(d) {
		if i >= len(p) || d[i] != p[i] {
			return nil
		}
		i++
	}
	for dir != nil && i < len(p) {
		dir = dir.lookupLocal(p[i])
		i++
	}
	return dir
}


// DirEntry describes a directory entry. The Depth and Height values
// are useful for presenting an entry in an indented fashion.
//
type DirEntry struct {
	Depth    int    // >= 0
	Height   int    // = DirList.MaxHeight - Depth, > 0
	Path     string // includes Name, relative to DirList root
	Name     string
	Synopsis string
}


type DirList struct {
	MaxHeight int // directory tree height, > 0
	List      []DirEntry
}


// listing creates a (linear) directory listing from a directory tree.
// If skipRoot is set, the root directory itself is excluded from the list.
//
func (root *Directory) listing(skipRoot bool) *DirList {
	if root == nil {
		return nil
	}

	// determine number of entries n and maximum height
	n := 0
	minDepth := 1 << 30 // infinity
	maxDepth := 0
	for d := range root.iter(skipRoot) {
		n++
		if minDepth > d.Depth {
			minDepth = d.Depth
		}
		if maxDepth < d.Depth {
			maxDepth = d.Depth
		}
	}
	maxHeight := maxDepth - minDepth + 1

	if n == 0 {
		return nil
	}

	// create list
	list := make([]DirEntry, n)
	i := 0
	for d := range root.iter(skipRoot) {
		p := &list[i]
		p.Depth = d.Depth - minDepth
		p.Height = maxHeight - p.Depth
		// the path is relative to root.Path - remove the root.Path
		// prefix (the prefix should always be present but avoid
		// crashes and check)
		path := d.Path
		if strings.HasPrefix(d.Path, root.Path) {
			path = d.Path[len(root.Path):]
		}
		// remove trailing '/' if any - path must be relative
		if len(path) > 0 && path[0] == '/' {
			path = path[1:]
		}
		p.Path = path
		p.Name = d.Name
		p.Synopsis = d.Text
		i++
	}

	return &DirList{maxHeight, list}
}
