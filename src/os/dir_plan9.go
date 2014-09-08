// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"io"
	"syscall"
)

func (file *File) readdir(n int) ([]FileInfo, error) {
	// If this file has no dirinfo, create one.
	if file.dirinfo == nil {
		file.dirinfo = new(dirInfo)
	}
	d := file.dirinfo
	size := n
	if size <= 0 {
		size = 100
		n = -1
	}
	fi := make([]FileInfo, 0, size) // Empty with room to grow.
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
				return fi, &PathError{"readdir", file.name, err}
			}
			if nb < syscall.STATFIXLEN {
				return fi, &PathError{"readdir", file.name, syscall.ErrShortStat}
			}
		}

		// Get a record from the buffer.
		b := d.buf[d.bufp:]
		m := int(uint16(b[0])|uint16(b[1])<<8) + 2
		if m < syscall.STATFIXLEN {
			return fi, &PathError{"readdir", file.name, syscall.ErrShortStat}
		}

		dir, err := syscall.UnmarshalDir(b[:m])
		if err != nil {
			return fi, &PathError{"readdir", file.name, err}
		}
		fi = append(fi, fileInfoFromStat(dir))

		d.bufp += m
		n--
	}

	if n >= 0 && len(fi) == 0 {
		return fi, io.EOF
	}
	return fi, nil
}

func (file *File) readdirnames(n int) (names []string, err error) {
	fi, err := file.Readdir(n)
	names = make([]string, len(fi))
	for i := range fi {
		names[i] = fi[i].Name()
	}
	return
}
