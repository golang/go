// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code dealing with package directory trees.

package godoc

import (
	"bytes"
	"go/doc"
	"go/parser"
	"go/token"
	"log"
	"os"
	pathpkg "path"
	"strings"
)

// Conventional name for directories containing test data.
// Excluded from directory trees.
//
const testdataDirName = "testdata"

type Directory struct {
	Depth    int
	Path     string       // directory path; includes Name
	Name     string       // directory name
	HasPkg   bool         // true if the directory contains at least one package
	Synopsis string       // package documentation, if any
	Dirs     []*Directory // subdirectories
}

func isGoFile(fi os.FileInfo) bool {
	name := fi.Name()
	return !fi.IsDir() &&
		len(name) > 0 && name[0] != '.' && // ignore .files
		pathpkg.Ext(name) == ".go"
}

func isPkgFile(fi os.FileInfo) bool {
	return isGoFile(fi) &&
		!strings.HasSuffix(fi.Name(), "_test.go") // ignore test files
}

func isPkgDir(fi os.FileInfo) bool {
	name := fi.Name()
	return fi.IsDir() && len(name) > 0 &&
		name[0] != '_' && name[0] != '.' // ignore _files and .files
}

type treeBuilder struct {
	c        *Corpus
	maxDepth int
}

func (b *treeBuilder) newDirTree(fset *token.FileSet, path, name string, depth int) *Directory {
	if name == testdataDirName {
		return nil
	}

	if depth >= b.maxDepth {
		// return a dummy directory so that the parent directory
		// doesn't get discarded just because we reached the max
		// directory depth
		return &Directory{
			Depth: depth,
			Path:  path,
			Name:  name,
		}
	}

	var synopses [3]string // prioritized package documentation (0 == highest priority)

	show := true // show in package listing
	hasPkgFiles := false
	haveSummary := false

	if hook := b.c.SummarizePackage; hook != nil {
		if summary, show0, ok := hook(strings.TrimPrefix(path, "/src/")); ok {
			hasPkgFiles = true
			show = show0
			synopses[0] = summary
			haveSummary = true
		}
	}

	list, _ := b.c.fs.ReadDir(path)

	// determine number of subdirectories and if there are package files
	var dirchs []chan *Directory

	for _, d := range list {
		switch {
		case isPkgDir(d):
			ch := make(chan *Directory, 1)
			dirchs = append(dirchs, ch)
			go func(d os.FileInfo) {
				name := d.Name()
				ch <- b.newDirTree(fset, pathpkg.Join(path, name), name, depth+1)
			}(d)
		case !haveSummary && isPkgFile(d):
			// looks like a package file, but may just be a file ending in ".go";
			// don't just count it yet (otherwise we may end up with hasPkgFiles even
			// though the directory doesn't contain any real package files - was bug)
			// no "optimal" package synopsis yet; continue to collect synopses
			file, err := b.c.parseFile(fset, pathpkg.Join(path, d.Name()),
				parser.ParseComments|parser.PackageClauseOnly)
			if err == nil {
				hasPkgFiles = true
				if file.Doc != nil {
					// prioritize documentation
					i := -1
					switch file.Name.Name {
					case name:
						i = 0 // normal case: directory name matches package name
					case "main":
						i = 1 // directory contains a main package
					default:
						i = 2 // none of the above
					}
					if 0 <= i && i < len(synopses) && synopses[i] == "" {
						synopses[i] = doc.Synopsis(file.Doc.Text())
					}
				}
				haveSummary = synopses[0] != ""
			}
		}
	}

	// create subdirectory tree
	var dirs []*Directory
	for _, ch := range dirchs {
		if d := <-ch; d != nil {
			dirs = append(dirs, d)
		}
	}

	// if there are no package files and no subdirectories
	// containing package files, ignore the directory
	if !hasPkgFiles && len(dirs) == 0 {
		return nil
	}

	// select the highest-priority synopsis for the directory entry, if any
	synopsis := ""
	for _, synopsis = range synopses {
		if synopsis != "" {
			break
		}
	}

	return &Directory{
		Depth:    depth,
		Path:     path,
		Name:     name,
		HasPkg:   hasPkgFiles && show, // TODO(bradfitz): add proper Hide field?
		Synopsis: synopsis,
		Dirs:     dirs,
	}
}

// newDirectory creates a new package directory tree with at most maxDepth
// levels, anchored at root. The result tree is pruned such that it only
// contains directories that contain package files or that contain
// subdirectories containing package files (transitively). If a non-nil
// pathFilter is provided, directory paths additionally must be accepted
// by the filter (i.e., pathFilter(path) must be true). If a value >= 0 is
// provided for maxDepth, nodes at larger depths are pruned as well; they
// are assumed to contain package files even if their contents are not known
// (i.e., in this case the tree may contain directories w/o any package files).
//
func (c *Corpus) newDirectory(root string, maxDepth int) *Directory {
	// The root could be a symbolic link so use Stat not Lstat.
	d, err := c.fs.Stat(root)
	// If we fail here, report detailed error messages; otherwise
	// is is hard to see why a directory tree was not built.
	switch {
	case err != nil:
		log.Printf("newDirectory(%s): %s", root, err)
		return nil
	case root != "/" && !isPkgDir(d):
		log.Printf("newDirectory(%s): not a package directory", root)
		return nil
	case root == "/" && !d.IsDir():
		log.Printf("newDirectory(%s): not a directory", root)
		return nil
	}
	if maxDepth < 0 {
		maxDepth = 1e6 // "infinity"
	}
	b := treeBuilder{c, maxDepth}
	// the file set provided is only for local parsing, no position
	// information escapes and thus we don't need to save the set
	return b.newDirTree(token.NewFileSet(), root, d.Name(), 0)
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

func splitPath(p string) []string {
	p = strings.TrimPrefix(p, "/")
	if p == "" {
		return nil
	}
	return strings.Split(p, "/")
}

// lookup looks for the *Directory for a given path, relative to dir.
func (dir *Directory) lookup(path string) *Directory {
	d := splitPath(dir.Path)
	p := splitPath(path)
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
	Path     string // directory path; includes Name, relative to DirList root
	Name     string // directory name
	HasPkg   bool   // true if the directory contains at least one package
	Synopsis string // package documentation, if any
}

type DirList struct {
	MaxHeight int // directory tree height, > 0
	List      []DirEntry
}

// listing creates a (linear) directory listing from a directory tree.
// If skipRoot is set, the root directory itself is excluded from the list.
// If filter is set, only the directory entries whose paths match the filter
// are included.
//
func (root *Directory) listing(skipRoot bool, filter func(string) bool) *DirList {
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
	list := make([]DirEntry, 0, n)
	for d := range root.iter(skipRoot) {
		if filter != nil && !filter(d.Path) {
			continue
		}
		var p DirEntry
		p.Depth = d.Depth - minDepth
		p.Height = maxHeight - p.Depth
		// the path is relative to root.Path - remove the root.Path
		// prefix (the prefix should always be present but avoid
		// crashes and check)
		path := strings.TrimPrefix(d.Path, root.Path)
		// remove leading separator if any - path must be relative
		path = strings.TrimPrefix(path, "/")
		p.Path = path
		p.Name = d.Name
		p.HasPkg = d.HasPkg
		p.Synopsis = d.Synopsis
		list = append(list, p)
	}

	return &DirList{maxHeight, list}
}
