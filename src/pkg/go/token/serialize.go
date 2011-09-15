// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import (
	"gob"
	"io"
	"os"
)

type serializedFile struct {
	// fields correspond 1:1 to fields with same (lower-case) name in File
	Name  string
	Base  int
	Size  int
	Lines []int
	Infos []lineInfo
}

type serializedFileSet struct {
	Base  int
	Files []serializedFile
}

// Read reads the fileset from r into s; s must not be nil.
func (s *FileSet) Read(r io.Reader) os.Error {
	var ss serializedFileSet
	if err := gob.NewDecoder(r).Decode(&ss); err != nil {
		return err
	}

	s.mutex.Lock()
	s.base = ss.Base
	files := make([]*File, len(ss.Files))
	for i := 0; i < len(ss.Files); i++ {
		f := &ss.Files[i]
		files[i] = &File{s, f.Name, f.Base, f.Size, f.Lines, f.Infos}
	}
	s.files = files
	s.last = nil
	s.mutex.Unlock()

	return nil
}

// Write writes the fileset s to w.
func (s *FileSet) Write(w io.Writer) os.Error {
	var ss serializedFileSet

	s.mutex.Lock()
	ss.Base = s.base
	files := make([]serializedFile, len(s.files))
	for i, f := range s.files {
		files[i] = serializedFile{f.name, f.base, f.size, f.lines, f.infos}
	}
	ss.Files = files
	s.mutex.Unlock()

	return gob.NewEncoder(w).Encode(ss)
}
