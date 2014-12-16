// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A simulated Unix-like file system for use within NaCl.
//
// The simulation is not particularly tied to NaCl other than the reuse
// of NaCl's definition for the Stat_t structure.
//
// The file system need never be written to disk, so it is represented as
// in-memory Go data structures, never in a serialized form.
//
// TODO: Perhaps support symlinks, although they muck everything up.

package syscall

import (
	"sync"
	"unsafe"
)

// Provided by package runtime.
func now() (sec int64, nsec int32)

// An fsys is a file system.
// Since there is no I/O (everything is in memory),
// the global lock mu protects the whole file system state,
// and that's okay.
type fsys struct {
	mu   sync.Mutex
	root *inode                    // root directory
	cwd  *inode                    // process current directory
	inum uint64                    // number of inodes created
	dev  []func() (devFile, error) // table for opening devices
}

// A devFile is the implementation required of device files
// like /dev/null or /dev/random.
type devFile interface {
	pread([]byte, int64) (int, error)
	pwrite([]byte, int64) (int, error)
}

// An inode is a (possibly special) file in the file system.
type inode struct {
	Stat_t
	data []byte
	dir  []dirent
}

// A dirent describes a single directory entry.
type dirent struct {
	name  string
	inode *inode
}

// An fsysFile is the fileImpl implementation backed by the file system.
type fsysFile struct {
	defaultFileImpl
	fsys     *fsys
	inode    *inode
	openmode int
	offset   int64
	dev      devFile
}

// newFsys creates a new file system.
func newFsys() *fsys {
	fs := &fsys{}
	fs.mu.Lock()
	defer fs.mu.Unlock()
	ip := fs.newInode()
	ip.Mode = 0555 | S_IFDIR
	fs.dirlink(ip, ".", ip)
	fs.dirlink(ip, "..", ip)
	fs.cwd = ip
	fs.root = ip
	return fs
}

var fs = newFsys()
var fsinit = func() {}

func init() {
	// do not trigger loading of zipped file system here
	oldFsinit := fsinit
	defer func() { fsinit = oldFsinit }()
	fsinit = func() {}
	Mkdir("/dev", 0555)
	Mkdir("/tmp", 0777)
	mkdev("/dev/null", 0666, openNull)
	mkdev("/dev/random", 0444, openRandom)
	mkdev("/dev/urandom", 0444, openRandom)
	mkdev("/dev/zero", 0666, openZero)
	chdirEnv()
}

func chdirEnv() {
	pwd, ok := Getenv("NACLPWD")
	if ok {
		chdir(pwd)
	}
}

// Except where indicated otherwise, unexported methods on fsys
// expect fs.mu to have been locked by the caller.

// newInode creates a new inode.
func (fs *fsys) newInode() *inode {
	fs.inum++
	ip := &inode{
		Stat_t: Stat_t{
			Ino:     fs.inum,
			Blksize: 512,
		},
	}
	return ip
}

// atime sets ip.Atime to the current time.
func (fs *fsys) atime(ip *inode) {
	sec, nsec := now()
	ip.Atime, ip.AtimeNsec = sec, int64(nsec)
}

// mtime sets ip.Mtime to the current time.
func (fs *fsys) mtime(ip *inode) {
	sec, nsec := now()
	ip.Mtime, ip.MtimeNsec = sec, int64(nsec)
}

// dirlookup looks for an entry in the directory dp with the given name.
// It returns the directory entry and its index within the directory.
func (fs *fsys) dirlookup(dp *inode, name string) (de *dirent, index int, err error) {
	fs.atime(dp)
	for i := range dp.dir {
		de := &dp.dir[i]
		if de.name == name {
			fs.atime(de.inode)
			return de, i, nil
		}
	}
	return nil, 0, ENOENT
}

// dirlink adds to the directory dp an entry for name pointing at the inode ip.
// If dp already contains an entry for name, that entry is overwritten.
func (fs *fsys) dirlink(dp *inode, name string, ip *inode) {
	fs.mtime(dp)
	fs.atime(ip)
	ip.Nlink++
	for i := range dp.dir {
		if dp.dir[i].name == name {
			dp.dir[i] = dirent{name, ip}
			return
		}
	}
	dp.dir = append(dp.dir, dirent{name, ip})
	dp.dirSize()
}

func (dp *inode) dirSize() {
	dp.Size = int64(len(dp.dir)) * (8 + 8 + 2 + 256) // Dirent
}

// skipelem splits path into the first element and the remainder.
// the returned first element contains no slashes, and the returned
// remainder does not begin with a slash.
func skipelem(path string) (elem, rest string) {
	for len(path) > 0 && path[0] == '/' {
		path = path[1:]
	}
	if len(path) == 0 {
		return "", ""
	}
	i := 0
	for i < len(path) && path[i] != '/' {
		i++
	}
	elem, path = path[:i], path[i:]
	for len(path) > 0 && path[0] == '/' {
		path = path[1:]
	}
	return elem, path
}

// namei translates a file system path name into an inode.
// If parent is false, the returned ip corresponds to the given name, and elem is the empty string.
// If parent is true, the walk stops at the next-to-last element in the name,
// so that ip is the parent directory and elem is the final element in the path.
func (fs *fsys) namei(path string, parent bool) (ip *inode, elem string, err error) {
	// Reject NUL in name.
	for i := 0; i < len(path); i++ {
		if path[i] == '\x00' {
			return nil, "", EINVAL
		}
	}

	// Reject empty name.
	if path == "" {
		return nil, "", EINVAL
	}

	if path[0] == '/' {
		ip = fs.root
	} else {
		ip = fs.cwd
	}

	for len(path) > 0 && path[len(path)-1] == '/' {
		path = path[:len(path)-1]
	}

	for {
		elem, rest := skipelem(path)
		if elem == "" {
			if parent && ip.Mode&S_IFMT == S_IFDIR {
				return ip, ".", nil
			}
			break
		}
		if ip.Mode&S_IFMT != S_IFDIR {
			return nil, "", ENOTDIR
		}
		if len(elem) >= 256 {
			return nil, "", ENAMETOOLONG
		}
		if parent && rest == "" {
			// Stop one level early.
			return ip, elem, nil
		}
		de, _, err := fs.dirlookup(ip, elem)
		if err != nil {
			return nil, "", err
		}
		ip = de.inode
		path = rest
	}
	if parent {
		return nil, "", ENOTDIR
	}
	return ip, "", nil
}

// open opens or creates a file with the given name, open mode,
// and permission mode bits.
func (fs *fsys) open(name string, openmode int, mode uint32) (fileImpl, error) {
	dp, elem, err := fs.namei(name, true)
	if err != nil {
		return nil, err
	}
	var (
		ip  *inode
		dev devFile
	)
	de, _, err := fs.dirlookup(dp, elem)
	if err != nil {
		if openmode&O_CREATE == 0 {
			return nil, err
		}
		ip = fs.newInode()
		ip.Mode = mode
		fs.dirlink(dp, elem, ip)
		if ip.Mode&S_IFMT == S_IFDIR {
			fs.dirlink(ip, ".", ip)
			fs.dirlink(ip, "..", dp)
		}
	} else {
		ip = de.inode
		if openmode&(O_CREATE|O_EXCL) == O_CREATE|O_EXCL {
			return nil, EEXIST
		}
		if openmode&O_TRUNC != 0 {
			if ip.Mode&S_IFMT == S_IFDIR {
				return nil, EISDIR
			}
			ip.data = nil
		}
		if ip.Mode&S_IFMT == S_IFCHR {
			if ip.Rdev < 0 || ip.Rdev >= int64(len(fs.dev)) || fs.dev[ip.Rdev] == nil {
				return nil, ENODEV
			}
			dev, err = fs.dev[ip.Rdev]()
			if err != nil {
				return nil, err
			}
		}
	}

	switch openmode & O_ACCMODE {
	case O_WRONLY, O_RDWR:
		if ip.Mode&S_IFMT == S_IFDIR {
			return nil, EISDIR
		}
	}

	switch ip.Mode & S_IFMT {
	case S_IFDIR:
		if openmode&O_ACCMODE != O_RDONLY {
			return nil, EISDIR
		}

	case S_IFREG:
		// ok

	case S_IFCHR:
		// handled above

	default:
		// TODO: some kind of special file
		return nil, EPERM
	}

	f := &fsysFile{
		fsys:     fs,
		inode:    ip,
		openmode: openmode,
		dev:      dev,
	}
	if openmode&O_APPEND != 0 {
		f.offset = ip.Size
	}
	return f, nil
}

// fsysFile methods to implement fileImpl.

func (f *fsysFile) stat(st *Stat_t) error {
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	*st = f.inode.Stat_t
	return nil
}

func (f *fsysFile) read(b []byte) (int, error) {
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	n, err := f.preadLocked(b, f.offset)
	f.offset += int64(n)
	return n, err
}

func ReadDirent(fd int, buf []byte) (int, error) {
	f, err := fdToFsysFile(fd)
	if err != nil {
		return 0, err
	}
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	if f.inode.Mode&S_IFMT != S_IFDIR {
		return 0, EINVAL
	}
	n, err := f.preadLocked(buf, f.offset)
	f.offset += int64(n)
	return n, err
}

func (f *fsysFile) write(b []byte) (int, error) {
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	n, err := f.pwriteLocked(b, f.offset)
	f.offset += int64(n)
	return n, err
}

func (f *fsysFile) seek(offset int64, whence int) (int64, error) {
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	switch whence {
	case 1:
		offset += f.offset
	case 2:
		offset += f.inode.Size
	}
	if offset < 0 {
		return 0, EINVAL
	}
	if offset > f.inode.Size {
		return 0, EINVAL
	}
	f.offset = offset
	return offset, nil
}

func (f *fsysFile) pread(b []byte, offset int64) (int, error) {
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	return f.preadLocked(b, offset)
}

func (f *fsysFile) pwrite(b []byte, offset int64) (int, error) {
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	return f.pwriteLocked(b, offset)
}

func (f *fsysFile) preadLocked(b []byte, offset int64) (int, error) {
	if f.openmode&O_ACCMODE == O_WRONLY {
		return 0, EINVAL
	}
	if offset < 0 {
		return 0, EINVAL
	}
	if f.dev != nil {
		f.fsys.atime(f.inode)
		f.fsys.mu.Unlock()
		defer f.fsys.mu.Lock()
		return f.dev.pread(b, offset)
	}
	if offset > f.inode.Size {
		return 0, nil
	}
	if int64(len(b)) > f.inode.Size-offset {
		b = b[:f.inode.Size-offset]
	}

	if f.inode.Mode&S_IFMT == S_IFDIR {
		if offset%direntSize != 0 || len(b) != 0 && len(b) < direntSize {
			return 0, EINVAL
		}
		fs.atime(f.inode)
		n := 0
		for len(b) >= direntSize {
			src := f.inode.dir[int(offset/direntSize)]
			dst := (*Dirent)(unsafe.Pointer(&b[0]))
			dst.Ino = int64(src.inode.Ino)
			dst.Off = offset
			dst.Reclen = direntSize
			for i := range dst.Name {
				dst.Name[i] = 0
			}
			copy(dst.Name[:], src.name)
			n += direntSize
			offset += direntSize
			b = b[direntSize:]
		}
		return n, nil
	}

	fs.atime(f.inode)
	n := copy(b, f.inode.data[offset:])
	return n, nil
}

func (f *fsysFile) pwriteLocked(b []byte, offset int64) (int, error) {
	if f.openmode&O_ACCMODE == O_RDONLY {
		return 0, EINVAL
	}
	if offset < 0 {
		return 0, EINVAL
	}
	if f.dev != nil {
		f.fsys.atime(f.inode)
		f.fsys.mu.Unlock()
		defer f.fsys.mu.Lock()
		return f.dev.pwrite(b, offset)
	}
	if offset > f.inode.Size {
		return 0, EINVAL
	}
	f.fsys.mtime(f.inode)
	n := copy(f.inode.data[offset:], b)
	if n < len(b) {
		f.inode.data = append(f.inode.data, b[n:]...)
		f.inode.Size = int64(len(f.inode.data))
	}
	return len(b), nil
}

// Standard Unix system calls.

func Open(path string, openmode int, perm uint32) (fd int, err error) {
	fsinit()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	f, err := fs.open(path, openmode, perm&0777|S_IFREG)
	if err != nil {
		return -1, err
	}
	return newFD(f), nil
}

func Mkdir(path string, perm uint32) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	_, err := fs.open(path, O_CREATE|O_EXCL, perm&0777|S_IFDIR)
	return err
}

func Getcwd(buf []byte) (n int, err error) {
	// Force package os to default to the old algorithm using .. and directory reads.
	return 0, ENOSYS
}

func Stat(path string, st *Stat_t) error {
	fsinit()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	ip, _, err := fs.namei(path, false)
	if err != nil {
		return err
	}
	*st = ip.Stat_t
	return nil
}

func Lstat(path string, st *Stat_t) error {
	return Stat(path, st)
}

func unlink(path string, isdir bool) error {
	fsinit()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	dp, elem, err := fs.namei(path, true)
	if err != nil {
		return err
	}
	if elem == "." || elem == ".." {
		return EINVAL
	}
	de, _, err := fs.dirlookup(dp, elem)
	if err != nil {
		return err
	}
	if isdir {
		if de.inode.Mode&S_IFMT != S_IFDIR {
			return ENOTDIR
		}
		if len(de.inode.dir) != 2 {
			return ENOTEMPTY
		}
	} else {
		if de.inode.Mode&S_IFMT == S_IFDIR {
			return EISDIR
		}
	}
	de.inode.Nlink--
	*de = dp.dir[len(dp.dir)-1]
	dp.dir = dp.dir[:len(dp.dir)-1]
	dp.dirSize()
	return nil
}

func Unlink(path string) error {
	return unlink(path, false)
}

func Rmdir(path string) error {
	return unlink(path, true)
}

func Chmod(path string, mode uint32) error {
	fsinit()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	ip, _, err := fs.namei(path, false)
	if err != nil {
		return err
	}
	ip.Mode = ip.Mode&^0777 | mode&0777
	return nil
}

func Fchmod(fd int, mode uint32) error {
	f, err := fdToFsysFile(fd)
	if err != nil {
		return err
	}
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	f.inode.Mode = f.inode.Mode&^0777 | mode&0777
	return nil
}

func Chown(path string, uid, gid int) error {
	fsinit()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	ip, _, err := fs.namei(path, false)
	if err != nil {
		return err
	}
	ip.Uid = uint32(uid)
	ip.Gid = uint32(gid)
	return nil
}

func Fchown(fd int, uid, gid int) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	f, err := fdToFsysFile(fd)
	if err != nil {
		return err
	}
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	f.inode.Uid = uint32(uid)
	f.inode.Gid = uint32(gid)
	return nil
}

func Lchown(path string, uid, gid int) error {
	return Chown(path, uid, gid)
}

func UtimesNano(path string, ts []Timespec) error {
	if len(ts) != 2 {
		return EINVAL
	}
	fsinit()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	ip, _, err := fs.namei(path, false)
	if err != nil {
		return err
	}
	ip.Atime = ts[0].Sec
	ip.AtimeNsec = int64(ts[0].Nsec)
	ip.Mtime = ts[1].Sec
	ip.MtimeNsec = int64(ts[1].Nsec)
	return nil
}

func Link(path, link string) error {
	fsinit()
	ip, _, err := fs.namei(path, false)
	if err != nil {
		return err
	}
	dp, elem, err := fs.namei(link, true)
	if err != nil {
		return err
	}
	if ip.Mode&S_IFMT == S_IFDIR {
		return EPERM
	}
	fs.dirlink(dp, elem, ip)
	return nil
}

func Rename(from, to string) error {
	fsinit()
	fdp, felem, err := fs.namei(from, true)
	if err != nil {
		return err
	}
	fde, _, err := fs.dirlookup(fdp, felem)
	if err != nil {
		return err
	}
	tdp, telem, err := fs.namei(to, true)
	if err != nil {
		return err
	}
	fs.dirlink(tdp, telem, fde.inode)
	fde.inode.Nlink--
	*fde = fdp.dir[len(fdp.dir)-1]
	fdp.dir = fdp.dir[:len(fdp.dir)-1]
	fdp.dirSize()
	return nil
}

func (fs *fsys) truncate(ip *inode, length int64) error {
	if length > 1e9 || ip.Mode&S_IFMT != S_IFREG {
		return EINVAL
	}
	if length < int64(len(ip.data)) {
		ip.data = ip.data[:length]
	} else {
		data := make([]byte, length)
		copy(data, ip.data)
		ip.data = data
	}
	ip.Size = int64(len(ip.data))
	return nil
}

func Truncate(path string, length int64) error {
	fsinit()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	ip, _, err := fs.namei(path, false)
	if err != nil {
		return err
	}
	return fs.truncate(ip, length)
}

func Ftruncate(fd int, length int64) error {
	f, err := fdToFsysFile(fd)
	if err != nil {
		return err
	}
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	return f.fsys.truncate(f.inode, length)
}

func Chdir(path string) error {
	fsinit()
	return chdir(path)
}

func chdir(path string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	ip, _, err := fs.namei(path, false)
	if err != nil {
		return err
	}
	fs.cwd = ip
	return nil
}

func Fchdir(fd int) error {
	f, err := fdToFsysFile(fd)
	if err != nil {
		return err
	}
	f.fsys.mu.Lock()
	defer f.fsys.mu.Unlock()
	if f.inode.Mode&S_IFMT != S_IFDIR {
		return ENOTDIR
	}
	fs.cwd = f.inode
	return nil
}

func Readlink(path string, buf []byte) (n int, err error) {
	return 0, ENOSYS
}

func Symlink(path, link string) error {
	return ENOSYS
}

func Fsync(fd int) error {
	return nil
}

// Special devices.

func mkdev(path string, mode uint32, open func() (devFile, error)) error {
	f, err := fs.open(path, O_CREATE|O_RDONLY|O_EXCL, S_IFCHR|mode)
	if err != nil {
		return err
	}
	ip := f.(*fsysFile).inode
	ip.Rdev = int64(len(fs.dev))
	fs.dev = append(fs.dev, open)
	return nil
}

type nullFile struct{}

func openNull() (devFile, error)                               { return &nullFile{}, nil }
func (f *nullFile) close() error                               { return nil }
func (f *nullFile) pread(b []byte, offset int64) (int, error)  { return 0, nil }
func (f *nullFile) pwrite(b []byte, offset int64) (int, error) { return len(b), nil }

type zeroFile struct{}

func openZero() (devFile, error)                               { return &zeroFile{}, nil }
func (f *zeroFile) close() error                               { return nil }
func (f *zeroFile) pwrite(b []byte, offset int64) (int, error) { return len(b), nil }

func (f *zeroFile) pread(b []byte, offset int64) (int, error) {
	for i := range b {
		b[i] = 0
	}
	return len(b), nil
}

type randomFile struct {
	naclFD int
}

func openRandom() (devFile, error) {
	fd, err := openNamedService("SecureRandom", O_RDONLY)
	if err != nil {
		return nil, err
	}
	return &randomFile{naclFD: fd}, nil
}

func (f *randomFile) close() error {
	naclClose(f.naclFD)
	f.naclFD = -1
	return nil
}

func (f *randomFile) pread(b []byte, offset int64) (int, error) {
	return naclRead(f.naclFD, b)
}

func (f *randomFile) pwrite(b []byte, offset int64) (int, error) {
	return 0, EPERM
}

func fdToFsysFile(fd int) (*fsysFile, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return nil, err
	}
	impl := f.impl
	fsysf, ok := impl.(*fsysFile)
	if !ok {
		return nil, EINVAL
	}
	return fsysf, nil
}

// create creates a file in the file system with the given name, mode, time, and data.
// It is meant to be called when initializing the file system image.
func create(name string, mode uint32, sec int64, data []byte) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	f, err := fs.open(name, O_CREATE|O_EXCL, mode)
	if err != nil {
		if mode&S_IFMT == S_IFDIR {
			ip, _, err := fs.namei(name, false)
			if err == nil && (ip.Mode&S_IFMT) == S_IFDIR {
				return nil // directory already exists
			}
		}
		return err
	}
	ip := f.(*fsysFile).inode
	ip.Atime = sec
	ip.Mtime = sec
	ip.Ctime = sec
	if len(data) > 0 {
		ip.Size = int64(len(data))
		ip.data = data
	}
	return nil
}
