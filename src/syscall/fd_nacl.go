// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// File descriptor support for Native Client.
// We want to provide access to a broader range of (simulated) files than
// Native Client allows, so we maintain our own file descriptor table exposed
// to higher-level packages.

package syscall

import (
	"sync"
)

// files is the table indexed by a file descriptor.
var files struct {
	sync.RWMutex
	tab []*file
}

// A file is an open file, something with a file descriptor.
// A particular *file may appear in files multiple times, due to use of Dup or Dup2.
type file struct {
	fdref int      // uses in files.tab
	impl  fileImpl // underlying implementation
}

// A fileImpl is the implementation of something that can be a file.
type fileImpl interface {
	// Standard operations.
	// These can be called concurrently from multiple goroutines.
	stat(*Stat_t) error
	read([]byte) (int, error)
	write([]byte) (int, error)
	seek(int64, int) (int64, error)
	pread([]byte, int64) (int, error)
	pwrite([]byte, int64) (int, error)

	// Close is called when the last reference to a *file is removed
	// from the file descriptor table. It may be called concurrently
	// with active operations such as blocked read or write calls.
	close() error
}

// newFD adds impl to the file descriptor table,
// returning the new file descriptor.
// Like Unix, it uses the lowest available descriptor.
func newFD(impl fileImpl) int {
	files.Lock()
	defer files.Unlock()
	f := &file{impl: impl, fdref: 1}
	for fd, oldf := range files.tab {
		if oldf == nil {
			files.tab[fd] = f
			return fd
		}
	}
	fd := len(files.tab)
	files.tab = append(files.tab, f)
	return fd
}

// Install Native Client stdin, stdout, stderr.
func init() {
	newFD(&naclFile{naclFD: 0})
	newFD(&naclFile{naclFD: 1})
	newFD(&naclFile{naclFD: 2})
}

// fdToFile retrieves the *file corresponding to a file descriptor.
func fdToFile(fd int) (*file, error) {
	files.Lock()
	defer files.Unlock()
	if fd < 0 || fd >= len(files.tab) || files.tab[fd] == nil {
		return nil, EBADF
	}
	return files.tab[fd], nil
}

func Close(fd int) error {
	files.Lock()
	if fd < 0 || fd >= len(files.tab) || files.tab[fd] == nil {
		files.Unlock()
		return EBADF
	}
	f := files.tab[fd]
	files.tab[fd] = nil
	f.fdref--
	fdref := f.fdref
	files.Unlock()
	if fdref > 0 {
		return nil
	}
	return f.impl.close()
}

func CloseOnExec(fd int) {
	// nothing to do - no exec
}

func Dup(fd int) (int, error) {
	files.Lock()
	defer files.Unlock()
	if fd < 0 || fd >= len(files.tab) || files.tab[fd] == nil {
		return -1, EBADF
	}
	f := files.tab[fd]
	f.fdref++
	for newfd, oldf := range files.tab {
		if oldf == nil {
			files.tab[newfd] = f
			return newfd, nil
		}
	}
	newfd := len(files.tab)
	files.tab = append(files.tab, f)
	return newfd, nil
}

func Dup2(fd, newfd int) error {
	files.Lock()
	defer files.Unlock()
	if fd < 0 || fd >= len(files.tab) || files.tab[fd] == nil || newfd < 0 || newfd >= len(files.tab)+100 {
		files.Unlock()
		return EBADF
	}
	f := files.tab[fd]
	f.fdref++
	for cap(files.tab) <= newfd {
		files.tab = append(files.tab[:cap(files.tab)], nil)
	}
	oldf := files.tab[newfd]
	var oldfdref int
	if oldf != nil {
		oldf.fdref--
		oldfdref = oldf.fdref
	}
	files.tab[newfd] = f
	files.Unlock()
	if oldf != nil {
		if oldfdref == 0 {
			oldf.impl.close()
		}
	}
	return nil
}

func Fstat(fd int, st *Stat_t) error {
	f, err := fdToFile(fd)
	if err != nil {
		return err
	}
	return f.impl.stat(st)
}

func Read(fd int, b []byte) (int, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}
	return f.impl.read(b)
}

var zerobuf [0]byte

func Write(fd int, b []byte) (int, error) {
	if b == nil {
		// avoid nil in syscalls; nacl doesn't like that.
		b = zerobuf[:]
	}
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}
	return f.impl.write(b)
}

func Pread(fd int, b []byte, offset int64) (int, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}
	return f.impl.pread(b, offset)
}

func Pwrite(fd int, b []byte, offset int64) (int, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}
	return f.impl.pwrite(b, offset)
}

func Seek(fd int, offset int64, whence int) (int64, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}
	return f.impl.seek(offset, whence)
}

// defaulFileImpl implements fileImpl.
// It can be embedded to complete a partial fileImpl implementation.
type defaultFileImpl struct{}

func (*defaultFileImpl) close() error                      { return nil }
func (*defaultFileImpl) stat(*Stat_t) error                { return ENOSYS }
func (*defaultFileImpl) read([]byte) (int, error)          { return 0, ENOSYS }
func (*defaultFileImpl) write([]byte) (int, error)         { return 0, ENOSYS }
func (*defaultFileImpl) seek(int64, int) (int64, error)    { return 0, ENOSYS }
func (*defaultFileImpl) pread([]byte, int64) (int, error)  { return 0, ENOSYS }
func (*defaultFileImpl) pwrite([]byte, int64) (int, error) { return 0, ENOSYS }

// naclFile is the fileImpl implementation for a Native Client file descriptor.
type naclFile struct {
	defaultFileImpl
	naclFD int
}

func (f *naclFile) stat(st *Stat_t) error {
	return naclFstat(f.naclFD, st)
}

func (f *naclFile) read(b []byte) (int, error) {
	n, err := naclRead(f.naclFD, b)
	if err != nil {
		n = 0
	}
	return n, err
}

// implemented in package runtime, to add time header on playground
func naclWrite(fd int, b []byte) int

func (f *naclFile) write(b []byte) (int, error) {
	n := naclWrite(f.naclFD, b)
	if n < 0 {
		return 0, Errno(-n)
	}
	return n, nil
}

func (f *naclFile) seek(off int64, whence int) (int64, error) {
	old := off
	err := naclSeek(f.naclFD, &off, whence)
	if err != nil {
		return old, err
	}
	return off, nil
}

func (f *naclFile) prw(b []byte, offset int64, rw func([]byte) (int, error)) (int, error) {
	// NaCl has no pread; simulate with seek and hope for no races.
	old, err := f.seek(0, 1)
	if err != nil {
		return 0, err
	}
	if _, err := f.seek(offset, 0); err != nil {
		return 0, err
	}
	n, err := rw(b)
	f.seek(old, 0)
	return n, err
}

func (f *naclFile) pread(b []byte, offset int64) (int, error) {
	return f.prw(b, offset, f.read)
}

func (f *naclFile) pwrite(b []byte, offset int64) (int, error) {
	return f.prw(b, offset, f.write)
}

func (f *naclFile) close() error {
	err := naclClose(f.naclFD)
	f.naclFD = -1
	return err
}

// A pipeFile is an in-memory implementation of a pipe.
// The byteq implementation is in net_nacl.go.
type pipeFile struct {
	defaultFileImpl
	rd *byteq
	wr *byteq
}

func (f *pipeFile) close() error {
	if f.rd != nil {
		f.rd.close()
	}
	if f.wr != nil {
		f.wr.close()
	}
	return nil
}

func (f *pipeFile) read(b []byte) (int, error) {
	if f.rd == nil {
		return 0, EINVAL
	}
	n, err := f.rd.read(b, 0)
	if err == EAGAIN {
		err = nil
	}
	return n, err
}

func (f *pipeFile) write(b []byte) (int, error) {
	if f.wr == nil {
		return 0, EINVAL
	}
	n, err := f.wr.write(b, 0)
	if err == EAGAIN {
		err = EPIPE
	}
	return n, err
}

func Pipe(fd []int) error {
	q := newByteq()
	fd[0] = newFD(&pipeFile{rd: q})
	fd[1] = newFD(&pipeFile{wr: q})
	return nil
}
