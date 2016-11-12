// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux

package interp

import "syscall"

func init() {
	for k, v := range map[string]externalFn{
		"os.Pipe":              ext۰os۰Pipe,
		"syscall.Close":        ext۰syscall۰Close,
		"syscall.Exit":         ext۰syscall۰Exit,
		"syscall.Fchown":       ext۰syscall۰Fchown,
		"syscall.Fstat":        ext۰syscall۰Fstat,
		"syscall.Ftruncate":    ext۰syscall۰Ftruncate,
		"syscall.Getpid":       ext۰syscall۰Getpid,
		"syscall.Getwd":        ext۰syscall۰Getwd,
		"syscall.Kill":         ext۰syscall۰Kill,
		"syscall.Link":         ext۰syscall۰Link,
		"syscall.Lstat":        ext۰syscall۰Lstat,
		"syscall.Mkdir":        ext۰syscall۰Mkdir,
		"syscall.Open":         ext۰syscall۰Open,
		"syscall.ParseDirent":  ext۰syscall۰ParseDirent,
		"syscall.RawSyscall":   ext۰syscall۰RawSyscall,
		"syscall.Read":         ext۰syscall۰Read,
		"syscall.ReadDirent":   ext۰syscall۰ReadDirent,
		"syscall.Readlink":     ext۰syscall۰Readlink,
		"syscall.Rmdir":        ext۰syscall۰Rmdir,
		"syscall.Seek":         ext۰syscall۰Seek,
		"syscall.Stat":         ext۰syscall۰Stat,
		"syscall.Symlink":      ext۰syscall۰Symlink,
		"syscall.Write":        ext۰syscall۰Write,
		"syscall.Unlink":       ext۰syscall۰Unlink,
		"syscall۰UtimesNano":   ext۰syscall۰UtimesNano,
		"syscall.setenv_c":     ext۰nop,
		"syscall.unsetenv_c":   ext۰nop,
		"syscall.runtime_envs": ext۰runtime۰environ,
	} {
		externals[k] = v
	}

	syswrite = syscall.Write
}

func ext۰os۰Pipe(fr *frame, args []value) value {
	// func os.Pipe() (r *File, w *File, err error)

	// The portable POSIX pipe(2) call is good enough for our needs.
	var p [2]int
	if err := syscall.Pipe(p[:]); err != nil {
		// TODO(adonovan): fix: return an *os.SyscallError.
		return tuple{nil, nil, wrapError(err)}
	}

	NewFile := fr.i.prog.ImportedPackage("os").Func("NewFile")
	r := call(fr.i, fr, 0, NewFile, []value{uintptr(p[0]), "|0"})
	w := call(fr.i, fr, 0, NewFile, []value{uintptr(p[1]), "|1"})
	return tuple{r, w, wrapError(nil)}
}

// overridden on darwin
var fillStat = func(st *syscall.Stat_t, stat structure) {
	stat[0] = st.Dev
	stat[1] = st.Ino
	stat[2] = st.Nlink
	stat[3] = st.Mode
	stat[4] = st.Uid
	stat[5] = st.Gid
	stat[7] = st.Rdev
	stat[8] = st.Size
	stat[9] = st.Blksize
	stat[10] = st.Blocks
	// TODO(adonovan): fix: copy Timespecs.
	// stat[11] = st.Atim
	// stat[12] = st.Mtim
	// stat[13] = st.Ctim
}

func ext۰syscall۰Close(fr *frame, args []value) value {
	// func Close(fd int) (err error)
	return wrapError(syscall.Close(args[0].(int)))
}

func ext۰syscall۰Exit(fr *frame, args []value) value {
	panic(exitPanic(args[0].(int)))
}

func ext۰syscall۰Fchown(fr *frame, args []value) value {
	fd := args[0].(int)
	uid := args[1].(int)
	gid := args[2].(int)
	return wrapError(syscall.Fchown(fd, uid, gid))
}

func ext۰syscall۰Fstat(fr *frame, args []value) value {
	// func Fstat(fd int, stat *Stat_t) (err error)
	fd := args[0].(int)
	stat := (*args[1].(*value)).(structure)

	var st syscall.Stat_t
	err := syscall.Fstat(fd, &st)
	fillStat(&st, stat)
	return wrapError(err)
}

func ext۰syscall۰Ftruncate(fr *frame, args []value) value {
	fd := args[0].(int)
	length := args[1].(int64)
	return wrapError(syscall.Ftruncate(fd, length))
}

func ext۰syscall۰Getpid(fr *frame, args []value) value {
	return syscall.Getpid()
}

func ext۰syscall۰Getwd(fr *frame, args []value) value {
	s, err := syscall.Getwd()
	return tuple{s, wrapError(err)}
}

func ext۰syscall۰Kill(fr *frame, args []value) value {
	// func Kill(pid int, sig Signal) (err error)
	return wrapError(syscall.Kill(args[0].(int), syscall.Signal(args[1].(int))))
}

func ext۰syscall۰Link(fr *frame, args []value) value {
	path := args[0].(string)
	link := args[1].(string)
	return wrapError(syscall.Link(path, link))
}

func ext۰syscall۰Lstat(fr *frame, args []value) value {
	// func Lstat(name string, stat *Stat_t) (err error)
	name := args[0].(string)
	stat := (*args[1].(*value)).(structure)

	var st syscall.Stat_t
	err := syscall.Lstat(name, &st)
	fillStat(&st, stat)
	return wrapError(err)
}

func ext۰syscall۰Mkdir(fr *frame, args []value) value {
	path := args[0].(string)
	mode := args[1].(uint32)
	return wrapError(syscall.Mkdir(path, mode))
}

func ext۰syscall۰Open(fr *frame, args []value) value {
	// func Open(path string, mode int, perm uint32) (fd int, err error) {
	path := args[0].(string)
	mode := args[1].(int)
	perm := args[2].(uint32)
	fd, err := syscall.Open(path, mode, perm)
	return tuple{fd, wrapError(err)}
}

func ext۰syscall۰ParseDirent(fr *frame, args []value) value {
	// func ParseDirent(buf []byte, max int, names []string) (consumed int, count int, newnames []string)
	max := args[1].(int)
	var names []string
	for _, iname := range args[2].([]value) {
		names = append(names, iname.(string))
	}
	consumed, count, newnames := syscall.ParseDirent(valueToBytes(args[0]), max, names)
	var inewnames []value
	for _, newname := range newnames {
		inewnames = append(inewnames, newname)
	}
	return tuple{consumed, count, inewnames}
}

func ext۰syscall۰RawSyscall(fr *frame, args []value) value {
	return tuple{uintptr(0), uintptr(0), uintptr(syscall.ENOSYS)}
}

func ext۰syscall۰Read(fr *frame, args []value) value {
	// func Read(fd int, p []byte) (n int, err error)
	fd := args[0].(int)
	p := args[1].([]value)
	b := make([]byte, len(p))
	n, err := syscall.Read(fd, b)
	for i := 0; i < n; i++ {
		p[i] = b[i]
	}
	return tuple{n, wrapError(err)}
}

func ext۰syscall۰ReadDirent(fr *frame, args []value) value {
	// func ReadDirent(fd int, buf []byte) (n int, err error)
	fd := args[0].(int)
	p := args[1].([]value)
	b := make([]byte, len(p))
	n, err := syscall.ReadDirent(fd, b)
	for i := 0; i < n; i++ {
		p[i] = b[i]
	}
	return tuple{n, wrapError(err)}
}

func ext۰syscall۰Readlink(fr *frame, args []value) value {
	path := args[0].(string)
	buf := valueToBytes(args[1])
	n, err := syscall.Readlink(path, buf)
	return tuple{n, wrapError(err)}
}

func ext۰syscall۰Rmdir(fr *frame, args []value) value {
	return wrapError(syscall.Rmdir(args[0].(string)))
}

func ext۰syscall۰Seek(fr *frame, args []value) value {
	fd := args[0].(int)
	offset := args[1].(int64)
	whence := args[2].(int)
	new, err := syscall.Seek(fd, offset, whence)
	return tuple{new, wrapError(err)}
}

func ext۰syscall۰Stat(fr *frame, args []value) value {
	// func Stat(name string, stat *Stat_t) (err error)
	name := args[0].(string)
	stat := (*args[1].(*value)).(structure)

	var st syscall.Stat_t
	err := syscall.Stat(name, &st)
	fillStat(&st, stat)
	return wrapError(err)
}

func ext۰syscall۰Symlink(fr *frame, args []value) value {
	path := args[0].(string)
	link := args[1].(string)
	return wrapError(syscall.Symlink(path, link))
}

func ext۰syscall۰Unlink(fr *frame, args []value) value {
	return wrapError(syscall.Unlink(args[0].(string)))
}

func ext۰syscall۰UtimesNano(fr *frame, args []value) value {
	path := args[0].(string)
	var ts [2]syscall.Timespec
	err := syscall.UtimesNano(path, ts[:])
	// TODO(adonovan): copy the Timespecs into args[1]
	return wrapError(err)
}

func ext۰syscall۰Write(fr *frame, args []value) value {
	// func Write(fd int, p []byte) (n int, err error)
	n, err := write(args[0].(int), valueToBytes(args[1]))
	return tuple{n, wrapError(err)}
}
