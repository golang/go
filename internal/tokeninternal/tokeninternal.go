// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package tokeninternal provides access to some internal features of the token
// package.
package tokeninternal

import (
	"fmt"
	"go/token"
	"sort"
	"sync"
	"unsafe"
)

// GetLines returns the table of line-start offsets from a token.File.
func GetLines(file *token.File) []int {
	// token.File has a Lines method on Go 1.21 and later.
	if file, ok := (interface{})(file).(interface{ Lines() []int }); ok {
		return file.Lines()
	}

	// This declaration must match that of token.File.
	// This creates a risk of dependency skew.
	// For now we check that the size of the two
	// declarations is the same, on the (fragile) assumption
	// that future changes would add fields.
	type tokenFile119 struct {
		_     string
		_     int
		_     int
		mu    sync.Mutex // we're not complete monsters
		lines []int
		_     []struct{}
	}
	type tokenFile118 struct {
		_ *token.FileSet // deleted in go1.19
		tokenFile119
	}

	type uP = unsafe.Pointer
	switch unsafe.Sizeof(*file) {
	case unsafe.Sizeof(tokenFile118{}):
		var ptr *tokenFile118
		*(*uP)(uP(&ptr)) = uP(file)
		ptr.mu.Lock()
		defer ptr.mu.Unlock()
		return ptr.lines

	case unsafe.Sizeof(tokenFile119{}):
		var ptr *tokenFile119
		*(*uP)(uP(&ptr)) = uP(file)
		ptr.mu.Lock()
		defer ptr.mu.Unlock()
		return ptr.lines

	default:
		panic("unexpected token.File size")
	}
}

// AddExistingFiles adds the specified files to the FileSet if they
// are not already present. It panics if any pair of files in the
// resulting FileSet would overlap.
func AddExistingFiles(fset *token.FileSet, files []*token.File) {
	// Punch through the FileSet encapsulation.
	type tokenFileSet struct {
		// This type remained essentially consistent from go1.16 to go1.21.
		mutex sync.RWMutex
		base  int
		files []*token.File
		_     *token.File // changed to atomic.Pointer[token.File] in go1.19
	}

	// If the size of token.FileSet changes, this will fail to compile.
	const delta = int64(unsafe.Sizeof(tokenFileSet{})) - int64(unsafe.Sizeof(token.FileSet{}))
	var _ [-delta * delta]int

	type uP = unsafe.Pointer
	var ptr *tokenFileSet
	*(*uP)(uP(&ptr)) = uP(fset)
	ptr.mutex.Lock()
	defer ptr.mutex.Unlock()

	// Merge and sort.
	newFiles := append(ptr.files, files...)
	sort.Slice(newFiles, func(i, j int) bool {
		return newFiles[i].Base() < newFiles[j].Base()
	})

	// Reject overlapping files.
	// Discard adjacent identical files.
	out := newFiles[:0]
	for i, file := range newFiles {
		if i > 0 {
			prev := newFiles[i-1]
			if file == prev {
				continue
			}
			if prev.Base()+prev.Size()+1 > file.Base() {
				panic(fmt.Sprintf("file %s (%d-%d) overlaps with file %s (%d-%d)",
					prev.Name(), prev.Base(), prev.Base()+prev.Size(),
					file.Name(), file.Base(), file.Base()+file.Size()))
			}
		}
		out = append(out, file)
	}
	newFiles = out

	ptr.files = newFiles

	// Advance FileSet.Base().
	if len(newFiles) > 0 {
		last := newFiles[len(newFiles)-1]
		newBase := last.Base() + last.Size() + 1
		if ptr.base < newBase {
			ptr.base = newBase
		}
	}
}

// FileSetFor returns a new FileSet containing a sequence of new Files with
// the same base, size, and line as the input files, for use in APIs that
// require a FileSet.
//
// Precondition: the input files must be non-overlapping, and sorted in order
// of their Base.
func FileSetFor(files ...*token.File) *token.FileSet {
	fset := token.NewFileSet()
	for _, f := range files {
		f2 := fset.AddFile(f.Name(), f.Base(), f.Size())
		lines := GetLines(f)
		f2.SetLines(lines)
	}
	return fset
}

// CloneFileSet creates a new FileSet holding all files in fset. It does not
// create copies of the token.Files in fset: they are added to the resulting
// FileSet unmodified.
func CloneFileSet(fset *token.FileSet) *token.FileSet {
	var files []*token.File
	fset.Iterate(func(f *token.File) bool {
		files = append(files, f)
		return true
	})
	newFileSet := token.NewFileSet()
	AddExistingFiles(newFileSet, files)
	return newFileSet
}
