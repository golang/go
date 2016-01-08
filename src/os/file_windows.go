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

// File represents an open file descriptor.
type File struct {
	*file
}

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
	isConsole bool
	lastbits  []byte // first few bytes of the last incomplete rune in last write
	readbuf   []rune // input console buffer
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
	var m uint32
	if syscall.GetConsoleMode(f.fd, &m) == nil {
		f.isConsole = true
	}
	runtime.SetFinalizer(f.file, (*file).close)
	return f
}

// NewFile returns a new File with the given file descriptor and name.
func NewFile(fd uintptr, name string) *File {
	h := syscall.Handle(fd)
	if h == syscall.InvalidHandle {
		return nil
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
	r, e := syscall.Open(name, flag|syscall.O_CLOEXEC, syscallMode(perm))
	if e != nil {
		return nil, e
	}
	return NewFile(uintptr(r), name), nil
}

func openDir(name string) (file *File, err error) {
	var mask string
	if len(name) == 2 && name[1] == ':' { // it is a drive letter, like C:
		mask = name + `*`
	} else {
		mask = name + `\*`
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
		namep, e := syscall.UTF16PtrFromString(name)
		if e != nil {
			return nil, e
		}
		e = syscall.GetFileAttributesEx(namep, syscall.GetFileExInfoStandard, (*byte)(unsafe.Pointer(&fa)))
		if e != nil {
			return nil, e
		}
		if fa.FileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY == 0 {
			return nil, e
		}
		d.isempty = true
	}
	d.path = name
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
// or Create instead.  It opens the named file with specified flag
// (O_RDONLY etc.) and perm, (0666 etc.) if applicable.  If successful,
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
		e = syscall.FindClose(syscall.Handle(file.fd))
	} else {
		e = syscall.CloseHandle(syscall.Handle(file.fd))
	}
	var err error
	if e != nil {
		err = &PathError{"close", file.name, e}
	}
	file.fd = syscall.InvalidHandle // so it can't be closed again

	// no need for a finalizer anymore
	runtime.SetFinalizer(file, nil)
	return err
}

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
			e := syscall.FindNextFile(syscall.Handle(file.fd), d)
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
		name := string(syscall.UTF16ToString(d.FileName[0:]))
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

// readConsole reads utf16 characters from console File,
// encodes them into utf8 and stores them in buffer b.
// It returns the number of utf8 bytes read and an error, if any.
func (f *File) readConsole(b []byte) (n int, err error) {
	if len(b) == 0 {
		return 0, nil
	}
	if len(f.readbuf) == 0 {
		numBytes := len(b)
		// Windows  can't read bytes over max of int16.
		// Some versions of Windows can read even less.
		// See golang.org/issue/13697.
		if numBytes > 10000 {
			numBytes = 10000
		}
		mbytes := make([]byte, numBytes)
		var nmb uint32
		err := syscall.ReadFile(f.fd, mbytes, &nmb, nil)
		if err != nil {
			return 0, err
		}
		if nmb > 0 {
			var pmb *byte
			if len(b) > 0 {
				pmb = &mbytes[0]
			}
			acp := windows.GetACP()
			nwc, err := windows.MultiByteToWideChar(acp, 2, pmb, int32(nmb), nil, 0)
			if err != nil {
				return 0, err
			}
			wchars := make([]uint16, nwc)
			pwc := &wchars[0]
			nwc, err = windows.MultiByteToWideChar(acp, 2, pmb, int32(nmb), pwc, int32(nwc))
			if err != nil {
				return 0, err
			}
			f.readbuf = utf16.Decode(wchars[:nwc])
		}
	}
	for i, r := range f.readbuf {
		if utf8.RuneLen(r) > len(b) {
			f.readbuf = f.readbuf[i:]
			return n, nil
		}
		nr := utf8.EncodeRune(b, r)
		b = b[nr:]
		n += nr
	}
	f.readbuf = nil
	return n, nil
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
	curoffset, e := syscall.Seek(f.fd, 0, 1)
	if e != nil {
		return 0, e
	}
	defer syscall.Seek(f.fd, curoffset, 0)
	o := syscall.Overlapped{
		OffsetHigh: uint32(off >> 32),
		Offset:     uint32(off),
	}
	var done uint32
	e = syscall.ReadFile(syscall.Handle(f.fd), b, &done, &o)
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
	curoffset, e := syscall.Seek(f.fd, 0, 1)
	if e != nil {
		return 0, e
	}
	defer syscall.Seek(f.fd, curoffset, 0)
	o := syscall.Overlapped{
		OffsetHigh: uint32(off >> 32),
		Offset:     uint32(off),
	}
	var done uint32
	e = syscall.WriteFile(syscall.Handle(f.fd), b, &done, &o)
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
	p, e := syscall.UTF16PtrFromString(name)
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
			}
		}
	}
	return &PathError{"remove", name, e}
}

func rename(oldname, newname string) error {
	e := windows.Rename(oldname, newname)
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
	n, err := syscall.UTF16PtrFromString(newname)
	if err != nil {
		return &LinkError{"link", oldname, newname, err}
	}
	o, err := syscall.UTF16PtrFromString(oldname)
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

	n, err := syscall.UTF16PtrFromString(newname)
	if err != nil {
		return &LinkError{"symlink", oldname, newname, err}
	}
	o, err := syscall.UTF16PtrFromString(oldname)
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

func fromSlash(path string) string {
	// Replace each '/' with '\\' if present
	var pathbuf []byte
	var lastSlash int
	for i, b := range path {
		if b == '/' {
			if pathbuf == nil {
				pathbuf = make([]byte, len(path))
			}
			copy(pathbuf[lastSlash:], path[lastSlash:i])
			pathbuf[i] = '\\'
			lastSlash = i + 1
		}
	}
	if pathbuf == nil {
		return path
	}

	copy(pathbuf[lastSlash:], path[lastSlash:])
	return string(pathbuf)
}

func dirname(path string) string {
	vol := volumeName(path)
	i := len(path) - 1
	for i >= len(vol) && !IsPathSeparator(path[i]) {
		i--
	}
	dir := path[len(vol) : i+1]
	last := len(dir) - 1
	if last > 0 && IsPathSeparator(dir[last]) {
		dir = dir[:last]
	}
	if dir == "" {
		dir = "."
	}
	return vol + dir
}
