// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/windows"
	"io"
	"runtime"
	"sync"
	"syscall"
	"unicode/utf16"
	"unicode/utf8"
	"unsafe"
)

// file is the real representation of *File.
// The extra level of indirection ensures that no clients of os
// can overwrite this data, which could cause the finalizer
// to close the wrong file descriptor.
type file struct {
	fd      syscall.Handle
	name    string
	dirinfo *dirInfo   // nil unless directory being read
	l       sync.Mutex // used to implement windows pread/pwrite

	// only for console io
	isConsole      bool
	lastbits       []byte   // first few bytes of the last incomplete rune in last write
	readuint16     []uint16 // buffer to hold uint16s obtained with ReadConsole
	readbyte       []byte   // buffer to hold decoding of readuint16 from utf16 to utf8
	readbyteOffset int      // readbyte[readOffset:] is yet to be consumed with file.Read
}

// Fd returns the Windows handle referencing the open file.
// The handle is valid only until f.Close is called or f is garbage collected.
func (file *File) Fd() uintptr {
	if file == nil {
		return uintptr(syscall.InvalidHandle)
	}
	return uintptr(file.fd)
}

// newFile returns a new File with the given file handle and name.
// Unlike NewFile, it does not check that h is syscall.InvalidHandle.
func newFile(h syscall.Handle, name string) *File {
	f := &File{&file{fd: h, name: name}}
	runtime.SetFinalizer(f.file, (*file).close)
	return f
}

// newConsoleFile creates new File that will be used as console.
func newConsoleFile(h syscall.Handle, name string) *File {
	f := newFile(h, name)
	f.isConsole = true
	return f
}

// NewFile returns a new File with the given file descriptor and name.
func NewFile(fd uintptr, name string) *File {
	h := syscall.Handle(fd)
	if h == syscall.InvalidHandle {
		return nil
	}
	var m uint32
	if syscall.GetConsoleMode(h, &m) == nil {
		return newConsoleFile(h, name)
	}
	return newFile(h, name)
}

// Auxiliary information if the File describes a directory
type dirInfo struct {
	data     syscall.Win32finddata
	needdata bool
	path     string
	isempty  bool // set if FindFirstFile returns ERROR_FILE_NOT_FOUND
}

func epipecheck(file *File, e error) {
}

const DevNull = "NUL"

func (f *file) isdir() bool { return f != nil && f.dirinfo != nil }

func openFile(name string, flag int, perm FileMode) (file *File, err error) {
	r, e := syscall.Open(fixLongPath(name), flag|syscall.O_CLOEXEC, syscallMode(perm))
	if e != nil {
		return nil, e
	}
	return NewFile(uintptr(r), name), nil
}

func openDir(name string) (file *File, err error) {
	var mask string

	path := fixLongPath(name)

	if len(path) == 2 && path[1] == ':' || (len(path) > 0 && path[len(path)-1] == '\\') { // it is a drive letter, like C:
		mask = path + `*`
	} else {
		mask = path + `\*`
	}
	maskp, e := syscall.UTF16PtrFromString(mask)
	if e != nil {
		return nil, e
	}
	d := new(dirInfo)
	r, e := syscall.FindFirstFile(maskp, &d.data)
	if e != nil {
		// FindFirstFile returns ERROR_FILE_NOT_FOUND when
		// no matching files can be found. Then, if directory
		// exists, we should proceed.
		if e != syscall.ERROR_FILE_NOT_FOUND {
			return nil, e
		}
		var fa syscall.Win32FileAttributeData
		pathp, e := syscall.UTF16PtrFromString(path)
		if e != nil {
			return nil, e
		}
		e = syscall.GetFileAttributesEx(pathp, syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&fa)))
		if e != nil {
			return nil, e
		}
		if fa.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY == 0 {
			return nil, e
		}
		d.isempty = true
	}
	d.path = path
	if !isAbs(d.path) {
		d.path, e = syscall.FullPath(d.path)
		if e != nil {
			return nil, e
		}
	}
	f := newFile(r, name)
	f.dirinfo = d
	return f, nil
}

// OpenFile is the generalized open call; most users will use Open
// or Create instead. It opens the named file with specified flag
// (O_RDONLY etc.) and perm, (0666 etc.) if applicable. If successful,
// methods on the returned File can be used for I/O.
// If there is an error, it will be of type *PathError.
func OpenFile(name string, flag int, perm FileMode) (*File, error) {
	if name == "" {
		return nil, &PathError{"open", name, syscall.ENOENT}
	}
	r, errf := openFile(name, flag, perm)
	if errf == nil {
		return r, nil
	}
	r, errd := openDir(name)
	if errd == nil {
		if flag&O_WRONLY != 0 || flag&O_RDWR != 0 {
			r.Close()
			return nil, &PathError{"open", name, syscall.EISDIR}
		}
		return r, nil
	}
	return nil, &PathError{"open", name, errf}
}

// Close closes the File, rendering it unusable for I/O.
// It returns an error, if any.
func (file *File) Close() error {
	if file == nil {
		return ErrInvalid
	}
	return file.file.close()
}

func (file *file) close() error {
	if file == nil {
		return syscall.EINVAL
	}
	if file.isdir() && file.dirinfo.isempty {
		// "special" empty directories
		return nil
	}
	if file.fd == syscall.InvalidHandle {
		return syscall.EINVAL
	}
	var e error
	if file.isdir() {
		e = syscall.FindClose(file.fd)
	} else {
		e = syscall.CloseHandle(file.fd)
	}
	var err error
	if e != nil {
		err = &PathError{"close", file.name, e}
	}
	file.fd = badFd // so it can't be closed again

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

var readConsole = syscall.ReadConsole // changed for testing

// readConsole reads utf16 characters from console File,
// encodes them into utf8 and stores them in buffer b.
// It returns the number of utf8 bytes read and an error, if any.
func (f *File) readConsole(b []byte) (n int, err error) {
	if len(b) == 0 {
		return 0, nil
	}

	if f.readuint16 == nil {
		// Note: syscall.ReadConsole fails for very large buffers.
		// The limit is somewhere around (but not exactly) 16384.
		// Stay well below.
		f.readuint16 = make([]uint16, 0, 10000)
		f.readbyte = make([]byte, 0, 4*cap(f.readuint16))
	}

	for f.readbyteOffset >= len(f.readbyte) {
		n := cap(f.readuint16) - len(f.readuint16)
		if n > len(b) {
			n = len(b)
		}
		var nw uint32
		err := readConsole(f.fd, &f.readuint16[:len(f.readuint16)+1][len(f.readuint16)], uint32(n), &nw, nil)
		if err != nil {
			return 0, err
		}
		uint16s := f.readuint16[:len(f.readuint16)+int(nw)]
		f.readuint16 = f.readuint16[:0]
		buf := f.readbyte[:0]
		for i := 0; i < len(uint16s); i++ {
			r := rune(uint16s[i])
			if utf16.IsSurrogate(r) {
				if i+1 == len(uint16s) {
					if nw > 0 {
						// Save half surrogate pair for next time.
						f.readuint16 = f.readuint16[:1]
						f.readuint16[0] = uint16(r)
						break
					}
					r = utf8.RuneError
				} else {
					r = utf16.DecodeRune(r, rune(uint16s[i+1]))
					if r != utf8.RuneError {
						i++
					}
				}
			}
			n := utf8.EncodeRune(buf[len(buf):cap(buf)], r)
			buf = buf[:len(buf)+n]
		}
		f.readbyte = buf
		f.readbyteOffset = 0
		if nw == 0 {
			break
		}
	}

	src := f.readbyte[f.readbyteOffset:]
	var i int
	for i = 0; i < len(src) && i < len(b); i++ {
		x := src[i]
		if x == 0x1A { // Ctrl-Z
			if i == 0 {
				f.readbyteOffset++
			}
			break
		}
		b[i] = x
	}
	f.readbyteOffset += i
	return i, nil
}

// read reads up to len(b) bytes from the File.
// It returns the number of bytes read and an error, if any.
func (f *File) read(b []byte) (n int, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	if f.isConsole {
		return f.readConsole(b)
	}
	return fixCount(syscall.Read(f.fd, b))
}

// pread reads len(b) bytes from the File starting at byte offset off.
// It returns the number of bytes read and the error, if any.
// EOF is signaled by a zero count with err set to 0.
func (f *File) pread(b []byte, off int64) (n int, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	curoffset, e := syscall.Seek(f.fd, 0, io.SeekCurrent)
	if e != nil {
		return 0, e
	}
	defer syscall.Seek(f.fd, curoffset, io.SeekStart)
	o := syscall.Overlapped{
		OffsetHigh: uint32(off >> 32),
		Offset:     uint32(off),
	}
	var done uint32
	e = syscall.ReadFile(f.fd, b, &done, &o)
	if e != nil {
		if e == syscall.ERROR_HANDLE_EOF {
			// end of file
			return 0, nil
		}
		return 0, e
	}
	return int(done), nil
}

// writeConsole writes len(b) bytes to the console File.
// It returns the number of bytes written and an error, if any.
func (f *File) writeConsole(b []byte) (n int, err error) {
	n = len(b)
	runes := make([]rune, 0, 256)
	if len(f.lastbits) > 0 {
		b = append(f.lastbits, b...)
		f.lastbits = nil

	}
	for len(b) >= utf8.UTFMax || utf8.FullRune(b) {
		r, l := utf8.DecodeRune(b)
		runes = append(runes, r)
		b = b[l:]
	}
	if len(b) > 0 {
		f.lastbits = make([]byte, len(b))
		copy(f.lastbits, b)
	}
	// syscall.WriteConsole seems to fail, if given large buffer.
	// So limit the buffer to 16000 characters. This number was
	// discovered by experimenting with syscall.WriteConsole.
	const maxWrite = 16000
	for len(runes) > 0 {
		m := len(runes)
		if m > maxWrite {
			m = maxWrite
		}
		chunk := runes[:m]
		runes = runes[m:]
		uint16s := utf16.Encode(chunk)
		for len(uint16s) > 0 {
			var written uint32
			err = syscall.WriteConsole(f.fd, &uint16s[0], uint32(len(uint16s)), &written, nil)
			if err != nil {
				return 0, nil
			}
			uint16s = uint16s[written:]
		}
	}
	return n, nil
}

// write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
func (f *File) write(b []byte) (n int, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	if f.isConsole {
		return f.writeConsole(b)
	}
	return fixCount(syscall.Write(f.fd, b))
}

// pwrite writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
func (f *File) pwrite(b []byte, off int64) (n int, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	curoffset, e := syscall.Seek(f.fd, 0, io.SeekCurrent)
	if e != nil {
		return 0, e
	}
	defer syscall.Seek(f.fd, curoffset, io.SeekStart)
	o := syscall.Overlapped{
		OffsetHigh: uint32(off >> 32),
		Offset:     uint32(off),
	}
	var done uint32
	e = syscall.WriteFile(f.fd, b, &done, &o)
	if e != nil {
		return 0, e
	}
	return int(done), nil
}

// seek sets the offset for the next Read or Write on file to offset, interpreted
// according to whence: 0 means relative to the origin of the file, 1 means
// relative to the current offset, and 2 means relative to the end.
// It returns the new offset and an error, if any.
func (f *File) seek(offset int64, whence int) (ret int64, err error) {
	f.l.Lock()
	defer f.l.Unlock()
	return syscall.Seek(f.fd, offset, whence)
}

// Truncate changes the size of the named file.
// If the file is a symbolic link, it changes the size of the link's target.
func Truncate(name string, size int64) error {
	f, e := OpenFile(name, O_WRONLY|O_CREATE, 0666)
	if e != nil {
		return e
	}
	defer f.Close()
	e1 := f.Truncate(size)
	if e1 != nil {
		return e1
	}
	return nil
}

// Remove removes the named file or directory.
// If there is an error, it will be of type *PathError.
func Remove(name string) error {
	p, e := syscall.UTF16PtrFromString(fixLongPath(name))
	if e != nil {
		return &PathError{"remove", name, e}
	}

	// Go file interface forces us to know whether
	// name is a file or directory. Try both.
	e = syscall.DeleteFile(p)
	if e == nil {
		return nil
	}
	e1 := syscall.RemoveDirectory(p)
	if e1 == nil {
		return nil
	}

	// Both failed: figure out which error to return.
	if e1 != e {
		a, e2 := syscall.GetFileAttributes(p)
		if e2 != nil {
			e = e2
		} else {
			if a&syscall.FILE_ATTRIBUTE_DIRECTORY != 0 {
				e = e1
			} else if a&syscall.FILE_ATTRIBUTE_READONLY != 0 {
				if e1 = syscall.SetFileAttributes(p, a&^syscall.FILE_ATTRIBUTE_READONLY); e1 == nil {
					if e = syscall.DeleteFile(p); e == nil {
						return nil
					}
				}
			}
		}
	}
	return &PathError{"remove", name, e}
}

func rename(oldname, newname string) error {
	e := windows.Rename(fixLongPath(oldname), fixLongPath(newname))
	if e != nil {
		return &LinkError{"rename", oldname, newname, e}
	}
	return nil
}

// Pipe returns a connected pair of Files; reads from r return bytes written to w.
// It returns the files and an error, if any.
func Pipe() (r *File, w *File, err error) {
	var p [2]syscall.Handle

	// See ../syscall/exec.go for description of lock.
	syscall.ForkLock.RLock()
	e := syscall.Pipe(p[0:])
	if e != nil {
		syscall.ForkLock.RUnlock()
		return nil, nil, NewSyscallError("pipe", e)
	}
	syscall.CloseOnExec(p[0])
	syscall.CloseOnExec(p[1])
	syscall.ForkLock.RUnlock()

	return NewFile(uintptr(p[0]), "|0"), NewFile(uintptr(p[1]), "|1"), nil
}

// TempDir returns the default directory to use for temporary files.
func TempDir() string {
	n := uint32(syscall.MAX_PATH)
	for {
		b := make([]uint16, n)
		n, _ = syscall.GetTempPath(uint32(len(b)), &b[0])
		if n > uint32(len(b)) {
			continue
		}
		if n > 0 && b[n-1] == '\\' {
			n--
		}
		return string(utf16.Decode(b[:n]))
	}
}

// Link creates newname as a hard link to the oldname file.
// If there is an error, it will be of type *LinkError.
func Link(oldname, newname string) error {
	n, err := syscall.UTF16PtrFromString(fixLongPath(newname))
	if err != nil {
		return &LinkError{"link", oldname, newname, err}
	}
	o, err := syscall.UTF16PtrFromString(fixLongPath(oldname))
	if err != nil {
		return &LinkError{"link", oldname, newname, err}
	}
	err = syscall.CreateHardLink(n, o, 0)
	if err != nil {
		return &LinkError{"link", oldname, newname, err}
	}
	return nil
}

// Symlink creates newname as a symbolic link to oldname.
// If there is an error, it will be of type *LinkError.
func Symlink(oldname, newname string) error {
	// CreateSymbolicLink is not supported before Windows Vista
	if syscall.LoadCreateSymbolicLink() != nil {
		return &LinkError{"symlink", oldname, newname, syscall.EWINDOWS}
	}

	// '/' does not work in link's content
	oldname = fromSlash(oldname)

	// need the exact location of the oldname when its relative to determine if its a directory
	destpath := oldname
	if !isAbs(oldname) {
		destpath = dirname(newname) + `\` + oldname
	}

	fi, err := Lstat(destpath)
	isdir := err == nil && fi.IsDir()

	n, err := syscall.UTF16PtrFromString(fixLongPath(newname))
	if err != nil {
		return &LinkError{"symlink", oldname, newname, err}
	}
	o, err := syscall.UTF16PtrFromString(fixLongPath(oldname))
	if err != nil {
		return &LinkError{"symlink", oldname, newname, err}
	}

	var flags uint32
	if isdir {
		flags |= syscall.SYMBOLIC_LINK_FLAG_DIRECTORY
	}
	err = syscall.CreateSymbolicLink(n, o, flags)
	if err != nil {
		return &LinkError{"symlink", oldname, newname, err}
	}
	return nil
}

const badFd = syscall.InvalidHandle
