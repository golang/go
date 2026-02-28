// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"io"
	"runtime"
	"syscall"
)

func (file *File) readdir(n int, mode readdirMode) (names []string, dirents []DirEntry, infos []FileInfo, err error) {
	if !file.isdir() {
		return nil, nil, nil, &PathError{Op: "readdir", Path: file.name, Err: syscall.ENOTDIR}
	}
	wantAll := n <= 0
	if wantAll {
		n = -1
	}
	d := &file.dirinfo.data
	for n != 0 && !file.dirinfo.isempty {
		if file.dirinfo.needdata {
			e := file.pfd.FindNextFile(d)
			runtime.KeepAlive(file)
			if e != nil {
				if e == syscall.ERROR_NO_MORE_FILES {
					break
				} else {
					err = &PathError{Op: "FindNextFile", Path: file.name, Err: e}
					return
				}
			}
		}
		file.dirinfo.needdata = true
		name := syscall.UTF16ToString(d.FileName[0:])
		if name == "." || name == ".." { // Useless names
			continue
		}
		if mode == readdirName {
			names = append(names, name)
		} else {
			f := newFileStatFromWin32finddata(d)
			f.name = name
			f.path = file.dirinfo.path
			f.appendNameToPath = true
			if mode == readdirDirEntry {
				dirents = append(dirents, dirEntry{f})
			} else {
				infos = append(infos, f)
			}
		}
		n--
	}
	if !wantAll && len(names)+len(dirents)+len(infos) == 0 {
		return nil, nil, nil, io.EOF
	}
	return names, dirents, infos, nil
}

type dirEntry struct {
	fs *fileStat
}

func (de dirEntry) Name() string            { return de.fs.Name() }
func (de dirEntry) IsDir() bool             { return de.fs.IsDir() }
func (de dirEntry) Type() FileMode          { return de.fs.Mode().Type() }
func (de dirEntry) Info() (FileInfo, error) { return de.fs, nil }
