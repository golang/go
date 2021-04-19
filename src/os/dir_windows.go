// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"io"
	"syscall"
)

func (file *File) readdir(n int) (fi []FileInfo, err error) {
	if file == nil {
		return nil, syscall.EINVAL
	}
	if !file.isdir() {
		return nil, &PathError{"Readdir", file.name, syscall.ENOTDIR}
	}
	if !file.dirinfo.isempty && file.fd == syscall.InvalidHandle {
		return nil, syscall.EINVAL
	}
	wantAll := n <= 0
	size := n
	if wantAll {
		n = -1
		size = 100
	}
	fi = make([]FileInfo, 0, size) // Empty with room to grow.
	d := &file.dirinfo.data
	for n != 0 && !file.dirinfo.isempty {
		if file.dirinfo.needdata {
			e := syscall.FindNextFile(file.fd, d)
			if e != nil {
				if e == syscall.ERROR_NO_MORE_FILES {
					break
				} else {
					err = &PathError{"FindNextFile", file.name, e}
					if !wantAll {
						fi = nil
					}
					return
				}
			}
		}
		file.dirinfo.needdata = true
		name := syscall.UTF16ToString(d.FileName[0:])
		if name == "." || name == ".." { // Useless names
			continue
		}
		f := &fileStat{
			name: name,
			sys: syscall.Win32FileAttributeData{
				FileAttributes: d.FileAttributes,
				CreationTime:   d.CreationTime,
				LastAccessTime: d.LastAccessTime,
				LastWriteTime:  d.LastWriteTime,
				FileSizeHigh:   d.FileSizeHigh,
				FileSizeLow:    d.FileSizeLow,
			},
			path: file.dirinfo.path + `\` + name,
		}
		n--
		fi = append(fi, f)
	}
	if !wantAll && len(fi) == 0 {
		return fi, io.EOF
	}
	return fi, nil
}

func (file *File) readdirnames(n int) (names []string, err error) {
	fis, err := file.Readdir(n)
	names = make([]string, len(fis))
	for i, fi := range fis {
		names[i] = fi.Name()
	}
	return names, err
}
