// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import "slices"

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

// Read calls decode to deserialize a file set into s; s must not be nil.
func (s *FileSet) Read(decode func(any) error) error {
	var ss serializedFileSet
	if err := decode(&ss); err != nil {
		return err
	}

	s.mutex.Lock()
	s.base = ss.Base
	for _, f := range ss.Files {
		s.tree.add(&File{
			name:  f.Name,
			base:  f.Base,
			size:  f.Size,
			lines: f.Lines,
			infos: f.Infos,
		})
	}
	s.last.Store(nil)
	s.mutex.Unlock()

	return nil
}

// Write calls encode to serialize the file set s.
func (s *FileSet) Write(encode func(any) error) error {
	var ss serializedFileSet

	s.mutex.Lock()
	ss.Base = s.base
	var files []serializedFile
	for f := range s.tree.all() {
		f.mutex.Lock()
		files = append(files, serializedFile{
			Name:  f.name,
			Base:  f.base,
			Size:  f.size,
			Lines: slices.Clone(f.lines),
			Infos: slices.Clone(f.infos),
		})
		f.mutex.Unlock()
	}
	ss.Files = files
	s.mutex.Unlock()

	return encode(ss)
}
