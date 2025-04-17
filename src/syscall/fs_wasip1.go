// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package syscall

import (
	"internal/stringslite"
	"runtime"
	"structs"
	"unsafe"
)

func init() {
	// Try to set stdio to non-blocking mode before the os package
	// calls NewFile for each fd. NewFile queries the non-blocking flag
	// but doesn't change it, even if the runtime supports non-blocking
	// stdio. Since WebAssembly modules are single-threaded, blocking
	// system calls temporarily halt execution of the module. If the
	// runtime supports non-blocking stdio, the Go runtime is able to
	// use the WASI net poller to poll for read/write readiness and is
	// able to schedule goroutines while waiting.
	SetNonblock(0, true)
	SetNonblock(1, true)
	SetNonblock(2, true)
}

type uintptr32 = uint32
type size = uint32
type fdflags = uint32
type filesize = uint64
type filetype = uint8
type lookupflags = uint32
type oflags = uint32
type rights = uint64
type timestamp = uint64
type dircookie = uint64
type filedelta = int64
type fstflags = uint32

type iovec struct {
	_      structs.HostLayout
	buf    uintptr32
	bufLen size
}

const (
	LOOKUP_SYMLINK_FOLLOW = 0x00000001
)

const (
	OFLAG_CREATE    = 0x0001
	OFLAG_DIRECTORY = 0x0002
	OFLAG_EXCL      = 0x0004
	OFLAG_TRUNC     = 0x0008
)

const (
	FDFLAG_APPEND   = 0x0001
	FDFLAG_DSYNC    = 0x0002
	FDFLAG_NONBLOCK = 0x0004
	FDFLAG_RSYNC    = 0x0008
	FDFLAG_SYNC     = 0x0010
)

const (
	RIGHT_FD_DATASYNC = 1 << iota
	RIGHT_FD_READ
	RIGHT_FD_SEEK
	RIGHT_FDSTAT_SET_FLAGS
	RIGHT_FD_SYNC
	RIGHT_FD_TELL
	RIGHT_FD_WRITE
	RIGHT_FD_ADVISE
	RIGHT_FD_ALLOCATE
	RIGHT_PATH_CREATE_DIRECTORY
	RIGHT_PATH_CREATE_FILE
	RIGHT_PATH_LINK_SOURCE
	RIGHT_PATH_LINK_TARGET
	RIGHT_PATH_OPEN
	RIGHT_FD_READDIR
	RIGHT_PATH_READLINK
	RIGHT_PATH_RENAME_SOURCE
	RIGHT_PATH_RENAME_TARGET
	RIGHT_PATH_FILESTAT_GET
	RIGHT_PATH_FILESTAT_SET_SIZE
	RIGHT_PATH_FILESTAT_SET_TIMES
	RIGHT_FD_FILESTAT_GET
	RIGHT_FD_FILESTAT_SET_SIZE
	RIGHT_FD_FILESTAT_SET_TIMES
	RIGHT_PATH_SYMLINK
	RIGHT_PATH_REMOVE_DIRECTORY
	RIGHT_PATH_UNLINK_FILE
	RIGHT_POLL_FD_READWRITE
	RIGHT_SOCK_SHUTDOWN
	RIGHT_SOCK_ACCEPT
)

const (
	WHENCE_SET = 0
	WHENCE_CUR = 1
	WHENCE_END = 2
)

const (
	FILESTAT_SET_ATIM     = 0x0001
	FILESTAT_SET_ATIM_NOW = 0x0002
	FILESTAT_SET_MTIM     = 0x0004
	FILESTAT_SET_MTIM_NOW = 0x0008
)

const (
	// Despite the rights being defined as a 64 bits integer in the spec,
	// wasmtime crashes the program if we set any of the upper 32 bits.
	fullRights  = rights(^uint32(0))
	readRights  = rights(RIGHT_FD_READ | RIGHT_FD_READDIR)
	writeRights = rights(RIGHT_FD_DATASYNC | RIGHT_FD_WRITE | RIGHT_FD_ALLOCATE | RIGHT_PATH_FILESTAT_SET_SIZE)

	// Some runtimes have very strict expectations when it comes to which
	// rights can be enabled on files opened by path_open. The fileRights
	// constant is used as a mask to retain only bits for operations that
	// are supported on files.
	fileRights rights = RIGHT_FD_DATASYNC |
		RIGHT_FD_READ |
		RIGHT_FD_SEEK |
		RIGHT_FDSTAT_SET_FLAGS |
		RIGHT_FD_SYNC |
		RIGHT_FD_TELL |
		RIGHT_FD_WRITE |
		RIGHT_FD_ADVISE |
		RIGHT_FD_ALLOCATE |
		RIGHT_PATH_CREATE_DIRECTORY |
		RIGHT_PATH_CREATE_FILE |
		RIGHT_PATH_LINK_SOURCE |
		RIGHT_PATH_LINK_TARGET |
		RIGHT_PATH_OPEN |
		RIGHT_FD_READDIR |
		RIGHT_PATH_READLINK |
		RIGHT_PATH_RENAME_SOURCE |
		RIGHT_PATH_RENAME_TARGET |
		RIGHT_PATH_FILESTAT_GET |
		RIGHT_PATH_FILESTAT_SET_SIZE |
		RIGHT_PATH_FILESTAT_SET_TIMES |
		RIGHT_FD_FILESTAT_GET |
		RIGHT_FD_FILESTAT_SET_SIZE |
		RIGHT_FD_FILESTAT_SET_TIMES |
		RIGHT_PATH_SYMLINK |
		RIGHT_PATH_REMOVE_DIRECTORY |
		RIGHT_PATH_UNLINK_FILE |
		RIGHT_POLL_FD_READWRITE

	// Runtimes like wasmtime and wasmedge will refuse to open directories
	// if the rights requested by the application exceed the operations that
	// can be performed on a directory.
	dirRights rights = RIGHT_FD_SEEK |
		RIGHT_FDSTAT_SET_FLAGS |
		RIGHT_FD_SYNC |
		RIGHT_PATH_CREATE_DIRECTORY |
		RIGHT_PATH_CREATE_FILE |
		RIGHT_PATH_LINK_SOURCE |
		RIGHT_PATH_LINK_TARGET |
		RIGHT_PATH_OPEN |
		RIGHT_FD_READDIR |
		RIGHT_PATH_READLINK |
		RIGHT_PATH_RENAME_SOURCE |
		RIGHT_PATH_RENAME_TARGET |
		RIGHT_PATH_FILESTAT_GET |
		RIGHT_PATH_FILESTAT_SET_SIZE |
		RIGHT_PATH_FILESTAT_SET_TIMES |
		RIGHT_FD_FILESTAT_GET |
		RIGHT_FD_FILESTAT_SET_TIMES |
		RIGHT_PATH_SYMLINK |
		RIGHT_PATH_REMOVE_DIRECTORY |
		RIGHT_PATH_UNLINK_FILE
)

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-fd_closefd-fd---result-errno
//
//go:wasmimport wasi_snapshot_preview1 fd_close
//go:noescape
func fd_close(fd int32) Errno

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-fd_filestat_set_sizefd-fd-size-filesize---result-errno
//
//go:wasmimport wasi_snapshot_preview1 fd_filestat_set_size
//go:noescape
func fd_filestat_set_size(fd int32, set_size filesize) Errno

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-fd_preadfd-fd-iovs-iovec_array-offset-filesize---resultsize-errno
//
//go:wasmimport wasi_snapshot_preview1 fd_pread
//go:noescape
func fd_pread(fd int32, iovs *iovec, iovsLen size, offset filesize, nread *size) Errno

//go:wasmimport wasi_snapshot_preview1 fd_pwrite
//go:noescape
func fd_pwrite(fd int32, iovs *iovec, iovsLen size, offset filesize, nwritten *size) Errno

//go:wasmimport wasi_snapshot_preview1 fd_read
//go:noescape
func fd_read(fd int32, iovs *iovec, iovsLen size, nread *size) Errno

//go:wasmimport wasi_snapshot_preview1 fd_readdir
//go:noescape
func fd_readdir(fd int32, buf *byte, bufLen size, cookie dircookie, nwritten *size) Errno

//go:wasmimport wasi_snapshot_preview1 fd_seek
//go:noescape
func fd_seek(fd int32, offset filedelta, whence uint32, newoffset *filesize) Errno

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-fd_fdstat_set_rightsfd-fd-fs_rights_base-rights-fs_rights_inheriting-rights---result-errno
//
//go:wasmimport wasi_snapshot_preview1 fd_fdstat_set_rights
//go:noescape
func fd_fdstat_set_rights(fd int32, rightsBase rights, rightsInheriting rights) Errno

//go:wasmimport wasi_snapshot_preview1 fd_filestat_get
//go:noescape
func fd_filestat_get(fd int32, buf unsafe.Pointer) Errno

//go:wasmimport wasi_snapshot_preview1 fd_write
//go:noescape
func fd_write(fd int32, iovs *iovec, iovsLen size, nwritten *size) Errno

//go:wasmimport wasi_snapshot_preview1 fd_sync
//go:noescape
func fd_sync(fd int32) Errno

//go:wasmimport wasi_snapshot_preview1 path_create_directory
//go:noescape
func path_create_directory(fd int32, path *byte, pathLen size) Errno

//go:wasmimport wasi_snapshot_preview1 path_filestat_get
//go:noescape
func path_filestat_get(fd int32, flags lookupflags, path *byte, pathLen size, buf unsafe.Pointer) Errno

//go:wasmimport wasi_snapshot_preview1 path_filestat_set_times
//go:noescape
func path_filestat_set_times(fd int32, flags lookupflags, path *byte, pathLen size, atim timestamp, mtim timestamp, fstflags fstflags) Errno

//go:wasmimport wasi_snapshot_preview1 path_link
//go:noescape
func path_link(oldFd int32, oldFlags lookupflags, oldPath *byte, oldPathLen size, newFd int32, newPath *byte, newPathLen size) Errno

//go:wasmimport wasi_snapshot_preview1 path_readlink
//go:noescape
func path_readlink(fd int32, path *byte, pathLen size, buf *byte, bufLen size, nwritten *size) Errno

//go:wasmimport wasi_snapshot_preview1 path_remove_directory
//go:noescape
func path_remove_directory(fd int32, path *byte, pathLen size) Errno

//go:wasmimport wasi_snapshot_preview1 path_rename
//go:noescape
func path_rename(oldFd int32, oldPath *byte, oldPathLen size, newFd int32, newPath *byte, newPathLen size) Errno

//go:wasmimport wasi_snapshot_preview1 path_symlink
//go:noescape
func path_symlink(oldPath *byte, oldPathLen size, fd int32, newPath *byte, newPathLen size) Errno

//go:wasmimport wasi_snapshot_preview1 path_unlink_file
//go:noescape
func path_unlink_file(fd int32, path *byte, pathLen size) Errno

//go:wasmimport wasi_snapshot_preview1 path_open
//go:noescape
func path_open(rootFD int32, dirflags lookupflags, path *byte, pathLen size, oflags oflags, fsRightsBase rights, fsRightsInheriting rights, fsFlags fdflags, fd *int32) Errno

//go:wasmimport wasi_snapshot_preview1 random_get
//go:noescape
func random_get(buf *byte, bufLen size) Errno

// https://github.com/WebAssembly/WASI/blob/a2b96e81c0586125cc4dc79a5be0b78d9a059925/legacy/preview1/docs.md#-fdstat-record
// fdflags must be at offset 2, hence the uint16 type rather than the
// fdflags (uint32) type.
type fdstat struct {
	_                structs.HostLayout
	filetype         filetype
	fdflags          uint16
	rightsBase       rights
	rightsInheriting rights
}

//go:wasmimport wasi_snapshot_preview1 fd_fdstat_get
//go:noescape
func fd_fdstat_get(fd int32, buf *fdstat) Errno

//go:wasmimport wasi_snapshot_preview1 fd_fdstat_set_flags
//go:noescape
func fd_fdstat_set_flags(fd int32, flags fdflags) Errno

// fd_fdstat_get_flags is accessed from internal/syscall/unix
//go:linkname fd_fdstat_get_flags

func fd_fdstat_get_flags(fd int) (uint32, error) {
	var stat fdstat
	errno := fd_fdstat_get(int32(fd), &stat)
	return uint32(stat.fdflags), errnoErr(errno)
}

// fd_fdstat_get_type is accessed from net
//go:linkname fd_fdstat_get_type

func fd_fdstat_get_type(fd int) (uint8, error) {
	var stat fdstat
	errno := fd_fdstat_get(int32(fd), &stat)
	return stat.filetype, errnoErr(errno)
}

type preopentype = uint8

const (
	preopentypeDir preopentype = iota
)

type prestatDir struct {
	_         structs.HostLayout
	prNameLen size
}

type prestat struct {
	_   structs.HostLayout
	typ preopentype
	dir prestatDir
}

//go:wasmimport wasi_snapshot_preview1 fd_prestat_get
//go:noescape
func fd_prestat_get(fd int32, prestat *prestat) Errno

//go:wasmimport wasi_snapshot_preview1 fd_prestat_dir_name
//go:noescape
func fd_prestat_dir_name(fd int32, path *byte, pathLen size) Errno

type opendir struct {
	fd   int32
	name string
}

// List of preopen directories that were exposed by the runtime. The first one
// is assumed to the be root directory of the file system, and others are seen
// as mount points at sub paths of the root.
var preopens []opendir

// Current working directory. We maintain this as a string and resolve paths in
// the code because wasmtime does not allow relative path lookups outside of the
// scope of a directory; a previous approach we tried consisted in maintaining
// open a file descriptor to the current directory so we could perform relative
// path lookups from that location, but it resulted in breaking path resolution
// from the current directory to its parent.
var cwd string

func init() {
	dirNameBuf := make([]byte, 256)
	// We start looking for preopens at fd=3 because 0, 1, and 2 are reserved
	// for standard input and outputs.
	for preopenFd := int32(3); ; preopenFd++ {
		var prestat prestat

		errno := fd_prestat_get(preopenFd, &prestat)
		if errno == EBADF {
			break
		}
		if errno == ENOTDIR || prestat.typ != preopentypeDir {
			continue
		}
		if errno != 0 {
			panic("fd_prestat: " + errno.Error())
		}
		if int(prestat.dir.prNameLen) > len(dirNameBuf) {
			dirNameBuf = make([]byte, prestat.dir.prNameLen)
		}

		errno = fd_prestat_dir_name(preopenFd, &dirNameBuf[0], prestat.dir.prNameLen)
		if errno != 0 {
			panic("fd_prestat_dir_name: " + errno.Error())
		}

		preopens = append(preopens, opendir{
			fd:   preopenFd,
			name: string(dirNameBuf[:prestat.dir.prNameLen]),
		})
	}

	if cwd, _ = Getenv("PWD"); cwd != "" {
		cwd = joinPath("/", cwd)
	} else if len(preopens) > 0 {
		cwd = preopens[0].name
	}
}

// Provided by package runtime.
func now() (sec int64, nsec int32)

//go:nosplit
func appendCleanPath(buf []byte, path string, lookupParent bool) ([]byte, bool) {
	i := 0
	for i < len(path) {
		for i < len(path) && path[i] == '/' {
			i++
		}

		j := i
		for j < len(path) && path[j] != '/' {
			j++
		}

		s := path[i:j]
		i = j

		switch s {
		case "":
			continue
		case ".":
			continue
		case "..":
			if !lookupParent {
				k := len(buf)
				for k > 0 && buf[k-1] != '/' {
					k--
				}
				for k > 1 && buf[k-1] == '/' {
					k--
				}
				buf = buf[:k]
				if k == 0 {
					lookupParent = true
				} else {
					s = ""
					continue
				}
			}
		default:
			lookupParent = false
		}

		if len(buf) > 0 && buf[len(buf)-1] != '/' {
			buf = append(buf, '/')
		}
		buf = append(buf, s...)
	}
	return buf, lookupParent
}

// joinPath concatenates dir and file paths, producing a cleaned path where
// "." and ".." have been removed, unless dir is relative and the references
// to parent directories in file represented a location relative to a parent
// of dir.
//
// This function is used for path resolution of all wasi functions expecting
// a path argument; the returned string is heap allocated, which we may want
// to optimize in the future. Instead of returning a string, the function
// could append the result to an output buffer that the functions in this
// file can manage to have allocated on the stack (e.g. initializing to a
// fixed capacity). Since it will significantly increase code complexity,
// we prefer to optimize for readability and maintainability at this time.
func joinPath(dir, file string) string {
	buf := make([]byte, 0, len(dir)+len(file)+1)
	if isAbs(dir) {
		buf = append(buf, '/')
	}
	buf, lookupParent := appendCleanPath(buf, dir, false)
	buf, _ = appendCleanPath(buf, file, lookupParent)
	// The appendCleanPath function cleans the path so it does not inject
	// references to the current directory. If both the dir and file args
	// were ".", this results in the output buffer being empty so we handle
	// this condition here.
	if len(buf) == 0 {
		buf = append(buf, '.')
	}
	// If the file ended with a '/' we make sure that the output also ends
	// with a '/'. This is needed to ensure that programs have a mechanism
	// to represent dereferencing symbolic links pointing to directories.
	if buf[len(buf)-1] != '/' && isDir(file) {
		buf = append(buf, '/')
	}
	return unsafe.String(&buf[0], len(buf))
}

func isAbs(path string) bool {
	return stringslite.HasPrefix(path, "/")
}

func isDir(path string) bool {
	return stringslite.HasSuffix(path, "/")
}

// preparePath returns the preopen file descriptor of the directory to perform
// path resolution from, along with the pair of pointer and length for the
// relative expression of path from the directory.
//
// If the path argument is not absolute, it is first appended to the current
// working directory before resolution.
func preparePath(path string) (int32, *byte, size) {
	var dirFd = int32(-1)
	var dirName string

	dir := "/"
	if !isAbs(path) {
		dir = cwd
	}
	path = joinPath(dir, path)

	for _, p := range preopens {
		if len(p.name) > len(dirName) && stringslite.HasPrefix(path, p.name) {
			dirFd, dirName = p.fd, p.name
		}
	}

	path = path[len(dirName):]
	for isAbs(path) {
		path = path[1:]
	}
	if len(path) == 0 {
		path = "."
	}

	return dirFd, unsafe.StringData(path), size(len(path))
}

func Open(path string, openmode int, perm uint32) (int, error) {
	if path == "" {
		return -1, EINVAL
	}
	dirFd, pathPtr, pathLen := preparePath(path)
	return openat(dirFd, pathPtr, pathLen, openmode, perm)
}

func Openat(dirFd int, path string, openmode int, perm uint32) (int, error) {
	return openat(int32(dirFd), unsafe.StringData(path), size(len(path)), openmode, perm)
}

func openat(dirFd int32, pathPtr *byte, pathLen size, openmode int, perm uint32) (int, error) {
	var oflags oflags
	if (openmode & O_CREATE) != 0 {
		oflags |= OFLAG_CREATE
	}
	if (openmode & O_TRUNC) != 0 {
		oflags |= OFLAG_TRUNC
	}
	if (openmode & O_EXCL) != 0 {
		oflags |= OFLAG_EXCL
	}

	var rights rights
	switch openmode & (O_RDONLY | O_WRONLY | O_RDWR) {
	case O_RDONLY:
		rights = fileRights & ^writeRights
	case O_WRONLY:
		rights = fileRights & ^readRights
	case O_RDWR:
		rights = fileRights
	}

	if (openmode & O_DIRECTORY) != 0 {
		if openmode&(O_WRONLY|O_RDWR) != 0 {
			return -1, EISDIR
		}
		oflags |= OFLAG_DIRECTORY
		rights &= dirRights
	}

	var fdflags fdflags
	if (openmode & O_APPEND) != 0 {
		fdflags |= FDFLAG_APPEND
	}
	if (openmode & O_SYNC) != 0 {
		fdflags |= FDFLAG_SYNC
	}

	var lflags lookupflags
	if openmode&O_NOFOLLOW == 0 {
		lflags = LOOKUP_SYMLINK_FOLLOW
	}

	var fd int32
	errno := path_open(
		dirFd,
		lflags,
		pathPtr,
		pathLen,
		oflags,
		rights,
		fileRights,
		fdflags,
		&fd,
	)
	if errno == EISDIR && oflags == 0 && fdflags == 0 && ((rights & writeRights) == 0) {
		// wasmtime and wasmedge will error if attempting to open a directory
		// because we are asking for too many rights. However, we cannot
		// determine ahead of time if the path we are about to open is a
		// directory, so instead we fallback to a second call to path_open with
		// a more limited set of rights.
		//
		// This approach is subject to a race if the file system is modified
		// concurrently, so we also inject OFLAG_DIRECTORY to ensure that we do
		// not accidentally open a file which is not a directory.
		errno = path_open(
			dirFd,
			LOOKUP_SYMLINK_FOLLOW,
			pathPtr,
			pathLen,
			oflags|OFLAG_DIRECTORY,
			rights&dirRights,
			fileRights,
			fdflags,
			&fd,
		)
	}
	return int(fd), errnoErr(errno)
}

func Close(fd int) error {
	errno := fd_close(int32(fd))
	return errnoErr(errno)
}

func CloseOnExec(fd int) {
	// nothing to do - no exec
}

func Mkdir(path string, perm uint32) error {
	if path == "" {
		return EINVAL
	}
	dirFd, pathPtr, pathLen := preparePath(path)
	errno := path_create_directory(dirFd, pathPtr, pathLen)
	return errnoErr(errno)
}

func ReadDir(fd int, buf []byte, cookie dircookie) (int, error) {
	var nwritten size
	errno := fd_readdir(int32(fd), &buf[0], size(len(buf)), cookie, &nwritten)
	return int(nwritten), errnoErr(errno)
}

type Stat_t struct {
	Dev      uint64
	Ino      uint64
	Filetype uint8
	Nlink    uint64
	Size     uint64
	Atime    uint64
	Mtime    uint64
	Ctime    uint64

	Mode int

	// Uid and Gid are always zero on wasip1 platforms
	Uid uint32
	Gid uint32
}

func Stat(path string, st *Stat_t) error {
	if path == "" {
		return EINVAL
	}
	dirFd, pathPtr, pathLen := preparePath(path)
	errno := path_filestat_get(dirFd, LOOKUP_SYMLINK_FOLLOW, pathPtr, pathLen, unsafe.Pointer(st))
	setDefaultMode(st)
	return errnoErr(errno)
}

func Lstat(path string, st *Stat_t) error {
	if path == "" {
		return EINVAL
	}
	dirFd, pathPtr, pathLen := preparePath(path)
	errno := path_filestat_get(dirFd, 0, pathPtr, pathLen, unsafe.Pointer(st))
	setDefaultMode(st)
	return errnoErr(errno)
}

func Fstat(fd int, st *Stat_t) error {
	errno := fd_filestat_get(int32(fd), unsafe.Pointer(st))
	setDefaultMode(st)
	return errnoErr(errno)
}

func setDefaultMode(st *Stat_t) {
	// WASI does not support unix-like permissions, but Go programs are likely
	// to expect the permission bits to not be zero so we set defaults to help
	// avoid breaking applications that are migrating to WASM.
	if st.Filetype == FILETYPE_DIRECTORY {
		st.Mode = 0700
	} else {
		st.Mode = 0600
	}
}

func Unlink(path string) error {
	if path == "" {
		return EINVAL
	}
	dirFd, pathPtr, pathLen := preparePath(path)
	errno := path_unlink_file(dirFd, pathPtr, pathLen)
	return errnoErr(errno)
}

func Rmdir(path string) error {
	if path == "" {
		return EINVAL
	}
	dirFd, pathPtr, pathLen := preparePath(path)
	errno := path_remove_directory(dirFd, pathPtr, pathLen)
	return errnoErr(errno)
}

func Chmod(path string, mode uint32) error {
	var stat Stat_t
	return Stat(path, &stat)
}

func Fchmod(fd int, mode uint32) error {
	var stat Stat_t
	return Fstat(fd, &stat)
}

func Chown(path string, uid, gid int) error {
	return ENOSYS
}

func Fchown(fd int, uid, gid int) error {
	return ENOSYS
}

func Lchown(path string, uid, gid int) error {
	return ENOSYS
}

func UtimesNano(path string, ts []Timespec) error {
	// UTIME_OMIT value must match internal/syscall/unix/at_wasip1.go
	const UTIME_OMIT = -0x2
	if path == "" {
		return EINVAL
	}
	dirFd, pathPtr, pathLen := preparePath(path)
	atime := TimespecToNsec(ts[0])
	mtime := TimespecToNsec(ts[1])
	if ts[0].Nsec == UTIME_OMIT || ts[1].Nsec == UTIME_OMIT {
		var st Stat_t
		if err := Stat(path, &st); err != nil {
			return err
		}
		if ts[0].Nsec == UTIME_OMIT {
			atime = int64(st.Atime)
		}
		if ts[1].Nsec == UTIME_OMIT {
			mtime = int64(st.Mtime)
		}
	}
	errno := path_filestat_set_times(
		dirFd,
		LOOKUP_SYMLINK_FOLLOW,
		pathPtr,
		pathLen,
		timestamp(atime),
		timestamp(mtime),
		FILESTAT_SET_ATIM|FILESTAT_SET_MTIM,
	)
	return errnoErr(errno)
}

func Rename(from, to string) error {
	if from == "" || to == "" {
		return EINVAL
	}
	oldDirFd, oldPathPtr, oldPathLen := preparePath(from)
	newDirFd, newPathPtr, newPathLen := preparePath(to)
	errno := path_rename(
		oldDirFd,
		oldPathPtr,
		oldPathLen,
		newDirFd,
		newPathPtr,
		newPathLen,
	)
	return errnoErr(errno)
}

func Truncate(path string, length int64) error {
	if path == "" {
		return EINVAL
	}
	fd, err := Open(path, O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer Close(fd)
	return Ftruncate(fd, length)
}

func Ftruncate(fd int, length int64) error {
	errno := fd_filestat_set_size(int32(fd), filesize(length))
	return errnoErr(errno)
}

const ImplementsGetwd = true

func Getwd() (string, error) {
	return cwd, nil
}

func Chdir(path string) error {
	if path == "" {
		return EINVAL
	}

	dir := "/"
	if !isAbs(path) {
		dir = cwd
	}
	path = joinPath(dir, path)

	var stat Stat_t
	dirFd, pathPtr, pathLen := preparePath(path)
	errno := path_filestat_get(dirFd, LOOKUP_SYMLINK_FOLLOW, pathPtr, pathLen, unsafe.Pointer(&stat))
	if errno != 0 {
		return errnoErr(errno)
	}
	if stat.Filetype != FILETYPE_DIRECTORY {
		return ENOTDIR
	}
	cwd = path
	return nil
}

func Readlink(path string, buf []byte) (n int, err error) {
	if path == "" {
		return 0, EINVAL
	}
	if len(buf) == 0 {
		return 0, nil
	}
	dirFd, pathPtr, pathLen := preparePath(path)
	var nwritten size
	errno := path_readlink(
		dirFd,
		pathPtr,
		pathLen,
		&buf[0],
		size(len(buf)),
		&nwritten,
	)
	// For some reason wasmtime returns ERANGE when the output buffer is
	// shorter than the symbolic link value. os.Readlink expects a nil
	// error and uses the fact that n is greater or equal to the buffer
	// length to assume that it needs to try again with a larger size.
	// This condition is handled in os.Readlink.
	return int(nwritten), errnoErr(errno)
}

func Link(path, link string) error {
	if path == "" || link == "" {
		return EINVAL
	}
	oldDirFd, oldPathPtr, oldPathLen := preparePath(path)
	newDirFd, newPathPtr, newPathLen := preparePath(link)
	errno := path_link(
		oldDirFd,
		0,
		oldPathPtr,
		oldPathLen,
		newDirFd,
		newPathPtr,
		newPathLen,
	)
	return errnoErr(errno)
}

func Symlink(path, link string) error {
	if path == "" || link == "" {
		return EINVAL
	}
	dirFd, pathPtr, pathlen := preparePath(link)
	errno := path_symlink(
		unsafe.StringData(path),
		size(len(path)),
		dirFd,
		pathPtr,
		pathlen,
	)
	return errnoErr(errno)
}

func Fsync(fd int) error {
	errno := fd_sync(int32(fd))
	return errnoErr(errno)
}

func makeIOVec(b []byte) *iovec {
	return &iovec{
		buf:    uintptr32(uintptr(unsafe.Pointer(unsafe.SliceData(b)))),
		bufLen: size(len(b)),
	}
}

func Read(fd int, b []byte) (int, error) {
	var nread size
	errno := fd_read(int32(fd), makeIOVec(b), 1, &nread)
	runtime.KeepAlive(b)
	return int(nread), errnoErr(errno)
}

func Write(fd int, b []byte) (int, error) {
	var nwritten size
	errno := fd_write(int32(fd), makeIOVec(b), 1, &nwritten)
	runtime.KeepAlive(b)
	return int(nwritten), errnoErr(errno)
}

func Pread(fd int, b []byte, offset int64) (int, error) {
	var nread size
	errno := fd_pread(int32(fd), makeIOVec(b), 1, filesize(offset), &nread)
	runtime.KeepAlive(b)
	return int(nread), errnoErr(errno)
}

func Pwrite(fd int, b []byte, offset int64) (int, error) {
	var nwritten size
	errno := fd_pwrite(int32(fd), makeIOVec(b), 1, filesize(offset), &nwritten)
	runtime.KeepAlive(b)
	return int(nwritten), errnoErr(errno)
}

func Seek(fd int, offset int64, whence int) (int64, error) {
	var newoffset filesize
	errno := fd_seek(int32(fd), filedelta(offset), uint32(whence), &newoffset)
	return int64(newoffset), errnoErr(errno)
}

func Dup(fd int) (int, error) {
	return 0, ENOSYS
}

func Dup2(fd, newfd int) error {
	return ENOSYS
}

func Pipe(fd []int) error {
	return ENOSYS
}

func RandomGet(b []byte) error {
	errno := random_get(unsafe.SliceData(b), size(len(b)))
	return errnoErr(errno)
}
