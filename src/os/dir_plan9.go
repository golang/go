// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"io"
	"io/fs"
	"syscall"
)

func (file *File) readdir(n int, mode readdirMode) (names []string, dirents []DirEntry, infos []FileInfo, err error) {
	var d *dirInfo
	for {
		d = file.dirinfo.Load()
		if d != nil {
			break
		}
		newD := new(dirInfo)
		if file.dirinfo.CompareAndSwap(nil, newD) {
			d = newD
			break
		}
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	size := n
	if size <= 0 {
		size = 100
		n = -1
	}
	for n != 0 {
		// Refill the buffer if necessary.
		if d.bufp >= d.nbuf {
			nb, err := file.Read(d.buf[:])

			// Update the buffer state before checking for errors.
			d.bufp, d.nbuf = 0, nb

			if err != nil {
				if err == io.EOF {
					break
				}
				return names, dirents, infos, &PathError{Op: "readdir", Path: file.name, Err: err}
			}
			if nb < syscall.STATFIXLEN {
				return names, dirents, infos, &PathError{Op: "readdir", Path: file.name, Err: syscall.ErrShortStat}
			}
		}

		// Get a record from the buffer.
		b := d.buf[d.bufp:]
		m := int(uint16(b[0])|uint16(b[1])<<8) + 2
		if m < syscall.STATFIXLEN {
			return names, dirents, infos, &PathError{Op: "readdir", Path: file.name, Err: syscall.ErrShortStat}
		}

		dir, err := syscall.UnmarshalDir(b[:m])
		if err != nil {
			return names, dirents, infos, &PathError{Op: "readdir", Path: file.name, Err: err}
		}

		if mode == readdirName {
			names = append(names, dir.Name)
		} else {
			f := fileInfoFromStat(dir)
			if mode == readdirDirEntry {
				dirents = append(dirents, dirEntry{f})
			} else {
				infos = append(infos, f)
			}
		}
		d.bufp += m
		n--
	}

	if n > 0 && len(names)+len(dirents)+len(infos) == 0 {
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

func (de dirEntry) String() string {
	return fs.FormatDirEntry(de)
}
