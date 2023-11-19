// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"internal/intern"
	"io"
	"io/fs"
	"os"
	"sync"
	"time"
)

var _ io.Reader = (*cyclicBuf)(nil)

type cyclicBuf struct {
	i   int
	buf []byte
}

func newCyclicBuf(s []byte) cyclicBuf {
	return cyclicBuf{buf: s}
}

// Read Implements the [io.Reader] interface to read cache contents
// after [io.EOF] is returned at the end of the read,
// the next read will start from the beginning
func (buf *cyclicBuf) Read(p []byte) (n int, err error) {
	if buf.Empty() { // If there is no readable data
		buf.i = 0 // Set the start of the read subscript to 0, and the next read can start from scratch
		return 0, nil
	}
	n = copy(p, buf.buf[buf.i:])
	buf.i += n
	return n, nil
}

func (buf *cyclicBuf) Seek(offset int64, whence int) (int64, error) {
	off := int(offset)
	switch whence {
	case io.SeekStart:
		buf.i = off
	case io.SeekCurrent:
		buf.i += off
	case io.SeekEnd:
		buf.i = len(buf.buf) - off
	}
	return int64(buf.i), nil
}

// Empty determines whether the cache is read to the end
func (buf *cyclicBuf) Empty() bool {
	return buf.i >= len(buf.buf)
}

// Copy dhallow copy oneself
func (buf *cyclicBuf) Copy() cyclicBuf {
	b := *buf
	return b
}

var dirOpenCache sync.Map //key is string value is *intern.Value[*dirOpenCacheEntry]

// TODO: wait go.dev/issue/62483 after , use unique.Handle instead

type dirOpenCacheEntry struct {
	path    string
	fd      *os.File
	modtime time.Time // The modification time when the cache is created
	buf     cyclicBuf
}

func newdirOpenCacheEntry(name string) (d *dirOpenCacheEntry, err error) {
	fd, fdinfo, err := openAndStat(name)
	if err != nil {
		return nil, err
	}
	modtime := fdinfo.ModTime() // Get change time
	var buf cyclicBuf
	if !fdinfo.IsDir() {
		file, err := io.ReadAll(fd)
		if err != nil {
			return nil, err
		}
		buf = newCyclicBuf(file)
	}
	ret := &dirOpenCacheEntry{path: name, fd: fd, modtime: modtime, buf: buf}
	return ret, nil
}

func (d *dirOpenCacheEntry) Readdir(count int) ([]fs.FileInfo, error) {
	ok, err := d.isNoRevise()
	if err != nil {
		return nil, err
	}
	if ok {
		return d.fd.Readdir(count)
	}
	d.resetRead()
	return d.Readdir(count)
}

func (d *dirOpenCacheEntry) Stat() (fs.FileInfo, error) {
	ok, err := d.isNoRevise()
	if err != nil {
		return nil, err
	}
	if ok {
		return d.fd.Stat()
	}
	d.resetRead()
	return d.Stat()
}

func (d *dirOpenCacheEntry) Close() error {
	if test {
		// Avoid temporary directory deletion failures and
		// Avoid test failures that rely on this assumption
		// Calling the Close method means closing at the system file handle
		dirOpenCache.Delete(d.path)
		d.fd.Close()
	}
	// After the cache is deleted true close
	return nil
}

// isNoRevise Returns whether the system file has not been modified
func (d *dirOpenCacheEntry) isNoRevise() (bool, error) {
	fdinfo, err := d.fd.Stat()
	if err != nil {
		return false, err
	}
	nowtime := fdinfo.ModTime()          // Get file modification time now
	return d.modtime.Equal(nowtime), nil // Compare the cache modification time with the current modification time to check whether there is no modification
}

// openAndStat call [os.Open] and [os.File.Stat]
func openAndStat(name string) (*os.File, fs.FileInfo, error) {
	fd, err := os.Open(name)
	if err != nil {
		return nil, nil, err
	}
	fdinfo, err := fd.Stat()
	if err != nil {
		return nil, nil, err
	}
	return fd, fdinfo, nil
}

func (d *dirOpenCacheEntry) resetRead() error {
	fd, fdinfo, err := openAndStat(d.path)
	if err != nil {
		return err
	}
	d.fd = fd
	d.modtime = fdinfo.ModTime()
	if !fdinfo.IsDir() {
		file, err := io.ReadAll(fd)
		if err != nil {
			return err
		}
		d.buf = newCyclicBuf(file)
	}
	dirOpenCache.CompareAndSwap(d.path, intern.Get(d), intern.Get(d.Copy()))
	return nil
}

// Seek implements the [io.Seeker] interface, which
// if not modified, moves the cache content offset
func (d *dirOpenCacheEntry) Seek(offset int64, whence int) (int64, error) {
	ok, err := d.isNoRevise()
	if err != nil {
		return 0, err
	}
	if ok {
		return d.buf.Seek(offset, whence)
	}
	// The file has been modified
	// recache and then move cache content offset
	err = d.resetRead()
	if err != nil {
		return 0, err
	}
	return d.Seek(offset, whence)
}

// Read implements the [io.Reader] interface
// and returns the cached content if it is not modified
func (d *dirOpenCacheEntry) Read(p []byte) (n int, err error) {
	ok, err := d.isNoRevise()
	if err != nil {
		return 0, err
	}
	if ok {
		return d.buf.Read(p)
	}
	// There are changes
	// re-cache and re-read
	err = d.resetRead()
	if err != nil {
		return 0, err
	}
	return d.Read(p)
}

// Copy make a shallow copy of yourself
func (d *dirOpenCacheEntry) Copy() *dirOpenCacheEntry {
	ret := *d
	ret.buf = ret.buf.Copy()
	return &ret
}
