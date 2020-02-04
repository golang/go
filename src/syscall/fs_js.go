// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package syscall

import (
	"errors"
	"io"
	"sync"
	"syscall/js"
)

// Provided by package runtime.
func now() (sec int64, nsec int32)

var jsProcess = js.Global().Get("process")
var jsFS = js.Global().Get("fs")
var constants = jsFS.Get("constants")

var uint8Array = js.Global().Get("Uint8Array")

var (
	nodeWRONLY = constants.Get("O_WRONLY").Int()
	nodeRDWR   = constants.Get("O_RDWR").Int()
	nodeCREATE = constants.Get("O_CREAT").Int()
	nodeTRUNC  = constants.Get("O_TRUNC").Int()
	nodeAPPEND = constants.Get("O_APPEND").Int()
	nodeEXCL   = constants.Get("O_EXCL").Int()
)

type jsFile struct {
	path    string
	entries []string
	dirIdx  int // entries[:dirIdx] have already been returned in ReadDirent
	pos     int64
	seeked  bool
}

var filesMu sync.Mutex
var files = map[int]*jsFile{
	0: {},
	1: {},
	2: {},
}

func fdToFile(fd int) (*jsFile, error) {
	filesMu.Lock()
	f, ok := files[fd]
	filesMu.Unlock()
	if !ok {
		return nil, EBADF
	}
	return f, nil
}

func Open(path string, openmode int, perm uint32) (int, error) {
	if err := checkPath(path); err != nil {
		return 0, err
	}

	flags := 0
	if openmode&O_WRONLY != 0 {
		flags |= nodeWRONLY
	}
	if openmode&O_RDWR != 0 {
		flags |= nodeRDWR
	}
	if openmode&O_CREATE != 0 {
		flags |= nodeCREATE
	}
	if openmode&O_TRUNC != 0 {
		flags |= nodeTRUNC
	}
	if openmode&O_APPEND != 0 {
		flags |= nodeAPPEND
	}
	if openmode&O_EXCL != 0 {
		flags |= nodeEXCL
	}
	if openmode&O_SYNC != 0 {
		return 0, errors.New("syscall.Open: O_SYNC is not supported by js/wasm")
	}

	jsFD, err := fsCall("open", path, flags, perm)
	if err != nil {
		return 0, err
	}
	fd := jsFD.Int()

	var entries []string
	if stat, err := fsCall("fstat", fd); err == nil && stat.Call("isDirectory").Bool() {
		dir, err := fsCall("readdir", path)
		if err != nil {
			return 0, err
		}
		entries = make([]string, dir.Length())
		for i := range entries {
			entries[i] = dir.Index(i).String()
		}
	}

	f := &jsFile{
		path:    path,
		entries: entries,
	}
	filesMu.Lock()
	files[fd] = f
	filesMu.Unlock()
	return fd, nil
}

func Close(fd int) error {
	filesMu.Lock()
	delete(files, fd)
	filesMu.Unlock()
	_, err := fsCall("close", fd)
	return err
}

func CloseOnExec(fd int) {
	// nothing to do - no exec
}

func Mkdir(path string, perm uint32) error {
	if err := checkPath(path); err != nil {
		return err
	}
	_, err := fsCall("mkdir", path, perm)
	return err
}

func ReadDirent(fd int, buf []byte) (int, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}
	if f.entries == nil {
		return 0, EINVAL
	}

	n := 0
	for f.dirIdx < len(f.entries) {
		entry := f.entries[f.dirIdx]
		l := 2 + len(entry)
		if l > len(buf) {
			break
		}
		buf[0] = byte(l)
		buf[1] = byte(l >> 8)
		copy(buf[2:], entry)
		buf = buf[l:]
		n += l
		f.dirIdx++
	}

	return n, nil
}

func setStat(st *Stat_t, jsSt js.Value) {
	st.Dev = int64(jsSt.Get("dev").Int())
	st.Ino = uint64(jsSt.Get("ino").Int())
	st.Mode = uint32(jsSt.Get("mode").Int())
	st.Nlink = uint32(jsSt.Get("nlink").Int())
	st.Uid = uint32(jsSt.Get("uid").Int())
	st.Gid = uint32(jsSt.Get("gid").Int())
	st.Rdev = int64(jsSt.Get("rdev").Int())
	st.Size = int64(jsSt.Get("size").Int())
	st.Blksize = int32(jsSt.Get("blksize").Int())
	st.Blocks = int32(jsSt.Get("blocks").Int())
	atime := int64(jsSt.Get("atimeMs").Int())
	st.Atime = atime / 1000
	st.AtimeNsec = (atime % 1000) * 1000000
	mtime := int64(jsSt.Get("mtimeMs").Int())
	st.Mtime = mtime / 1000
	st.MtimeNsec = (mtime % 1000) * 1000000
	ctime := int64(jsSt.Get("ctimeMs").Int())
	st.Ctime = ctime / 1000
	st.CtimeNsec = (ctime % 1000) * 1000000
}

func Stat(path string, st *Stat_t) error {
	if err := checkPath(path); err != nil {
		return err
	}
	jsSt, err := fsCall("stat", path)
	if err != nil {
		return err
	}
	setStat(st, jsSt)
	return nil
}

func Lstat(path string, st *Stat_t) error {
	if err := checkPath(path); err != nil {
		return err
	}
	jsSt, err := fsCall("lstat", path)
	if err != nil {
		return err
	}
	setStat(st, jsSt)
	return nil
}

func Fstat(fd int, st *Stat_t) error {
	jsSt, err := fsCall("fstat", fd)
	if err != nil {
		return err
	}
	setStat(st, jsSt)
	return nil
}

func Unlink(path string) error {
	if err := checkPath(path); err != nil {
		return err
	}
	_, err := fsCall("unlink", path)
	return err
}

func Rmdir(path string) error {
	if err := checkPath(path); err != nil {
		return err
	}
	_, err := fsCall("rmdir", path)
	return err
}

func Chmod(path string, mode uint32) error {
	if err := checkPath(path); err != nil {
		return err
	}
	_, err := fsCall("chmod", path, mode)
	return err
}

func Fchmod(fd int, mode uint32) error {
	_, err := fsCall("fchmod", fd, mode)
	return err
}

func Chown(path string, uid, gid int) error {
	if err := checkPath(path); err != nil {
		return err
	}
	_, err := fsCall("chown", path, uint32(uid), uint32(gid))
	return err
}

func Fchown(fd int, uid, gid int) error {
	_, err := fsCall("fchown", fd, uint32(uid), uint32(gid))
	return err
}

func Lchown(path string, uid, gid int) error {
	if err := checkPath(path); err != nil {
		return err
	}
	if jsFS.Get("lchown").IsUndefined() {
		// fs.lchown is unavailable on Linux until Node.js 10.6.0
		// TODO(neelance): remove when we require at least this Node.js version
		return ENOSYS
	}
	_, err := fsCall("lchown", path, uint32(uid), uint32(gid))
	return err
}

func UtimesNano(path string, ts []Timespec) error {
	if err := checkPath(path); err != nil {
		return err
	}
	if len(ts) != 2 {
		return EINVAL
	}
	atime := ts[0].Sec
	mtime := ts[1].Sec
	_, err := fsCall("utimes", path, atime, mtime)
	return err
}

func Rename(from, to string) error {
	if err := checkPath(from); err != nil {
		return err
	}
	if err := checkPath(to); err != nil {
		return err
	}
	_, err := fsCall("rename", from, to)
	return err
}

func Truncate(path string, length int64) error {
	if err := checkPath(path); err != nil {
		return err
	}
	_, err := fsCall("truncate", path, length)
	return err
}

func Ftruncate(fd int, length int64) error {
	_, err := fsCall("ftruncate", fd, length)
	return err
}

func Getcwd(buf []byte) (n int, err error) {
	defer recoverErr(&err)
	cwd := jsProcess.Call("cwd").String()
	n = copy(buf, cwd)
	return
}

func Chdir(path string) (err error) {
	if err := checkPath(path); err != nil {
		return err
	}
	defer recoverErr(&err)
	jsProcess.Call("chdir", path)
	return
}

func Fchdir(fd int) error {
	f, err := fdToFile(fd)
	if err != nil {
		return err
	}
	return Chdir(f.path)
}

func Readlink(path string, buf []byte) (n int, err error) {
	if err := checkPath(path); err != nil {
		return 0, err
	}
	dst, err := fsCall("readlink", path)
	if err != nil {
		return 0, err
	}
	n = copy(buf, dst.String())
	return n, nil
}

func Link(path, link string) error {
	if err := checkPath(path); err != nil {
		return err
	}
	if err := checkPath(link); err != nil {
		return err
	}
	_, err := fsCall("link", path, link)
	return err
}

func Symlink(path, link string) error {
	if err := checkPath(path); err != nil {
		return err
	}
	if err := checkPath(link); err != nil {
		return err
	}
	_, err := fsCall("symlink", path, link)
	return err
}

func Fsync(fd int) error {
	_, err := fsCall("fsync", fd)
	return err
}

func Read(fd int, b []byte) (int, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}

	if f.seeked {
		n, err := Pread(fd, b, f.pos)
		f.pos += int64(n)
		return n, err
	}

	buf := uint8Array.New(len(b))
	n, err := fsCall("read", fd, buf, 0, len(b), nil)
	if err != nil {
		return 0, err
	}
	js.CopyBytesToGo(b, buf)

	n2 := n.Int()
	f.pos += int64(n2)
	return n2, err
}

func Write(fd int, b []byte) (int, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}

	if f.seeked {
		n, err := Pwrite(fd, b, f.pos)
		f.pos += int64(n)
		return n, err
	}

	if faketime && (fd == 1 || fd == 2) {
		n := faketimeWrite(fd, b)
		if n < 0 {
			return 0, errnoErr(Errno(-n))
		}
		return n, nil
	}

	buf := uint8Array.New(len(b))
	js.CopyBytesToJS(buf, b)
	n, err := fsCall("write", fd, buf, 0, len(b), nil)
	if err != nil {
		return 0, err
	}
	n2 := n.Int()
	f.pos += int64(n2)
	return n2, err
}

func Pread(fd int, b []byte, offset int64) (int, error) {
	buf := uint8Array.New(len(b))
	n, err := fsCall("read", fd, buf, 0, len(b), offset)
	if err != nil {
		return 0, err
	}
	js.CopyBytesToGo(b, buf)
	return n.Int(), nil
}

func Pwrite(fd int, b []byte, offset int64) (int, error) {
	buf := uint8Array.New(len(b))
	js.CopyBytesToJS(buf, b)
	n, err := fsCall("write", fd, buf, 0, len(b), offset)
	if err != nil {
		return 0, err
	}
	return n.Int(), nil
}

func Seek(fd int, offset int64, whence int) (int64, error) {
	f, err := fdToFile(fd)
	if err != nil {
		return 0, err
	}

	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = f.pos + offset
	case io.SeekEnd:
		var st Stat_t
		if err := Fstat(fd, &st); err != nil {
			return 0, err
		}
		newPos = st.Size + offset
	default:
		return 0, errnoErr(EINVAL)
	}

	if newPos < 0 {
		return 0, errnoErr(EINVAL)
	}

	f.seeked = true
	f.dirIdx = 0 // Reset directory read position. See issue 35767.
	f.pos = newPos
	return newPos, nil
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

func fsCall(name string, args ...interface{}) (js.Value, error) {
	type callResult struct {
		val js.Value
		err error
	}

	c := make(chan callResult, 1)
	jsFS.Call(name, append(args, js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		var res callResult

		if len(args) >= 1 { // on Node.js 8, fs.utimes calls the callback without any arguments
			if jsErr := args[0]; !jsErr.IsNull() {
				res.err = mapJSError(jsErr)
			}
		}

		res.val = js.Undefined()
		if len(args) >= 2 {
			res.val = args[1]
		}

		c <- res
		return nil
	}))...)
	res := <-c
	return res.val, res.err
}

// checkPath checks that the path is not empty and that it contains no null characters.
func checkPath(path string) error {
	if path == "" {
		return EINVAL
	}
	for i := 0; i < len(path); i++ {
		if path[i] == '\x00' {
			return EINVAL
		}
	}
	return nil
}

func recoverErr(errPtr *error) {
	if err := recover(); err != nil {
		jsErr, ok := err.(js.Error)
		if !ok {
			panic(err)
		}
		*errPtr = mapJSError(jsErr.Value)
	}
}

// mapJSError maps an error given by Node.js to the appropriate Go error
func mapJSError(jsErr js.Value) error {
	errno, ok := errnoByCode[jsErr.Get("code").String()]
	if !ok {
		panic(jsErr)
	}
	return errnoErr(Errno(errno))
}
