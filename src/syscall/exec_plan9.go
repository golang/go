// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fork, exec, wait, etc.

package syscall

import (
	"internal/itoa"
	"runtime"
	"sync"
	"unsafe"
)

// ForkLock is not used on plan9.
var ForkLock sync.RWMutex

// gstringb reads a non-empty string from b, prefixed with a 16-bit length in little-endian order.
// It returns the string as a byte slice, or nil if b is too short to contain the length or
// the full string.
//
//go:nosplit
func gstringb(b []byte) []byte {
	if len(b) < 2 {
		return nil
	}
	n, b := gbit16(b)
	if int(n) > len(b) {
		return nil
	}
	return b[:n]
}

// Offset of the name field in a 9P directory entry - see UnmarshalDir() in dir_plan9.go
const nameOffset = 39

// gdirname returns the first filename from a buffer of directory entries,
// and a slice containing the remaining directory entries.
// If the buffer doesn't start with a valid directory entry, the returned name is nil.
//
//go:nosplit
func gdirname(buf []byte) (name []byte, rest []byte) {
	if len(buf) < 2 {
		return
	}
	size, buf := gbit16(buf)
	if size < STATFIXLEN || int(size) > len(buf) {
		return
	}
	name = gstringb(buf[nameOffset:size])
	rest = buf[size:]
	return
}

// StringSlicePtr converts a slice of strings to a slice of pointers
// to NUL-terminated byte arrays. If any string contains a NUL byte
// this function panics instead of returning an error.
//
// Deprecated: Use SlicePtrFromStrings instead.
func StringSlicePtr(ss []string) []*byte {
	bb := make([]*byte, len(ss)+1)
	for i := 0; i < len(ss); i++ {
		bb[i] = StringBytePtr(ss[i])
	}
	bb[len(ss)] = nil
	return bb
}

// SlicePtrFromStrings converts a slice of strings to a slice of
// pointers to NUL-terminated byte arrays. If any string contains
// a NUL byte, it returns (nil, EINVAL).
func SlicePtrFromStrings(ss []string) ([]*byte, error) {
	var err error
	bb := make([]*byte, len(ss)+1)
	for i := 0; i < len(ss); i++ {
		bb[i], err = BytePtrFromString(ss[i])
		if err != nil {
			return nil, err
		}
	}
	bb[len(ss)] = nil
	return bb, nil
}

// readdirnames returns the names of files inside the directory represented by dirfd.
func readdirnames(dirfd int) (names []string, err error) {
	names = make([]string, 0, 100)
	var buf [STATMAX]byte

	for {
		n, e := Read(dirfd, buf[:])
		if e != nil {
			return nil, e
		}
		if n == 0 {
			break
		}
		for b := buf[:n]; len(b) > 0; {
			var s []byte
			s, b = gdirname(b)
			if s == nil {
				return nil, ErrBadStat
			}
			names = append(names, string(s))
		}
	}
	return
}

// name of the directory containing names and control files for all open file descriptors
var dupdev, _ = BytePtrFromString("#d")

// forkAndExecInChild forks the process, calling dup onto 0..len(fd)
// and finally invoking exec(argv0, argvv, envv) in the child.
// If a dup or exec fails, it writes the error string to pipe.
// (The pipe write end is close-on-exec so if exec succeeds, it will be closed.)
//
// In the child, this function must not acquire any locks, because
// they might have been locked at the time of the fork. This means
// no rescheduling, no malloc calls, and no new stack segments.
// The calls to RawSyscall are okay because they are assembly
// functions that do not grow the stack.
//
//go:norace
func forkAndExecInChild(argv0 *byte, argv []*byte, envv []envItem, dir *byte, attr *ProcAttr, pipe int, rflag int) (pid int, err error) {
	// Declare all variables at top in case any
	// declarations require heap allocation (e.g., errbuf).
	var (
		r1       uintptr
		nextfd   int
		i        int
		clearenv int
		envfd    int
		errbuf   [ERRMAX]byte
		statbuf  [STATMAX]byte
		dupdevfd int
	)

	// Guard against side effects of shuffling fds below.
	// Make sure that nextfd is beyond any currently open files so
	// that we can't run the risk of overwriting any of them.
	fd := make([]int, len(attr.Files))
	nextfd = len(attr.Files)
	for i, ufd := range attr.Files {
		if nextfd < int(ufd) {
			nextfd = int(ufd)
		}
		fd[i] = int(ufd)
	}
	nextfd++

	if envv != nil {
		clearenv = RFCENVG
	}

	// About to call fork.
	// No more allocation or calls of non-assembly functions.
	r1, _, _ = RawSyscall(SYS_RFORK, uintptr(RFPROC|RFFDG|RFREND|clearenv|rflag), 0, 0)

	if r1 != 0 {
		if int32(r1) == -1 {
			return 0, NewError(errstr())
		}
		// parent; return PID
		return int(r1), nil
	}

	// Fork succeeded, now in child.

	// Close fds we don't need.
	r1, _, _ = RawSyscall(SYS_OPEN, uintptr(unsafe.Pointer(dupdev)), uintptr(O_RDONLY), 0)
	dupdevfd = int(r1)
	if dupdevfd == -1 {
		goto childerror
	}
dirloop:
	for {
		r1, _, _ = RawSyscall6(SYS_PREAD, uintptr(dupdevfd), uintptr(unsafe.Pointer(&statbuf[0])), uintptr(len(statbuf)), ^uintptr(0), ^uintptr(0), 0)
		n := int(r1)
		switch n {
		case -1:
			goto childerror
		case 0:
			break dirloop
		}
		for b := statbuf[:n]; len(b) > 0; {
			var s []byte
			s, b = gdirname(b)
			if s == nil {
				copy(errbuf[:], ErrBadStat.Error())
				goto childerror1
			}
			if s[len(s)-1] == 'l' {
				// control file for descriptor <N> is named <N>ctl
				continue
			}
			closeFdExcept(int(atoi(s)), pipe, dupdevfd, fd)
		}
	}
	RawSyscall(SYS_CLOSE, uintptr(dupdevfd), 0, 0)

	// Write new environment variables.
	if envv != nil {
		for i = 0; i < len(envv); i++ {
			r1, _, _ = RawSyscall(SYS_CREATE, uintptr(unsafe.Pointer(envv[i].name)), uintptr(O_WRONLY), uintptr(0666))

			if int32(r1) == -1 {
				goto childerror
			}

			envfd = int(r1)

			r1, _, _ = RawSyscall6(SYS_PWRITE, uintptr(envfd), uintptr(unsafe.Pointer(envv[i].value)), uintptr(envv[i].nvalue),
				^uintptr(0), ^uintptr(0), 0)

			if int32(r1) == -1 || int(r1) != envv[i].nvalue {
				goto childerror
			}

			r1, _, _ = RawSyscall(SYS_CLOSE, uintptr(envfd), 0, 0)

			if int32(r1) == -1 {
				goto childerror
			}
		}
	}

	// Chdir
	if dir != nil {
		r1, _, _ = RawSyscall(SYS_CHDIR, uintptr(unsafe.Pointer(dir)), 0, 0)
		if int32(r1) == -1 {
			goto childerror
		}
	}

	// Pass 1: look for fd[i] < i and move those up above len(fd)
	// so that pass 2 won't stomp on an fd it needs later.
	if pipe < nextfd {
		r1, _, _ = RawSyscall(SYS_DUP, uintptr(pipe), uintptr(nextfd), 0)
		if int32(r1) == -1 {
			goto childerror
		}
		pipe = nextfd
		nextfd++
	}
	for i = 0; i < len(fd); i++ {
		if fd[i] >= 0 && fd[i] < int(i) {
			if nextfd == pipe { // don't stomp on pipe
				nextfd++
			}
			r1, _, _ = RawSyscall(SYS_DUP, uintptr(fd[i]), uintptr(nextfd), 0)
			if int32(r1) == -1 {
				goto childerror
			}

			fd[i] = nextfd
			nextfd++
		}
	}

	// Pass 2: dup fd[i] down onto i.
	for i = 0; i < len(fd); i++ {
		if fd[i] == -1 {
			RawSyscall(SYS_CLOSE, uintptr(i), 0, 0)
			continue
		}
		if fd[i] == int(i) {
			continue
		}
		r1, _, _ = RawSyscall(SYS_DUP, uintptr(fd[i]), uintptr(i), 0)
		if int32(r1) == -1 {
			goto childerror
		}
	}

	// Pass 3: close fd[i] if it was moved in the previous pass.
	for i = 0; i < len(fd); i++ {
		if fd[i] >= 0 && fd[i] != int(i) {
			RawSyscall(SYS_CLOSE, uintptr(fd[i]), 0, 0)
		}
	}

	// Time to exec.
	r1, _, _ = RawSyscall(SYS_EXEC,
		uintptr(unsafe.Pointer(argv0)),
		uintptr(unsafe.Pointer(&argv[0])), 0)

childerror:
	// send error string on pipe
	RawSyscall(SYS_ERRSTR, uintptr(unsafe.Pointer(&errbuf[0])), uintptr(len(errbuf)), 0)
childerror1:
	errbuf[len(errbuf)-1] = 0
	i = 0
	for i < len(errbuf) && errbuf[i] != 0 {
		i++
	}

	RawSyscall6(SYS_PWRITE, uintptr(pipe), uintptr(unsafe.Pointer(&errbuf[0])), uintptr(i),
		^uintptr(0), ^uintptr(0), 0)

	for {
		RawSyscall(SYS_EXITS, 0, 0, 0)
	}
}

// close the numbered file descriptor, unless it is fd1, fd2, or a member of fds.
//
//go:nosplit
func closeFdExcept(n int, fd1 int, fd2 int, fds []int) {
	if n == fd1 || n == fd2 {
		return
	}
	for _, fd := range fds {
		if n == fd {
			return
		}
	}
	RawSyscall(SYS_CLOSE, uintptr(n), 0, 0)
}

func cexecPipe(p []int) error {
	e := Pipe(p)
	if e != nil {
		return e
	}

	fd, e := Open("#d/"+itoa.Itoa(p[1]), O_RDWR|O_CLOEXEC)
	if e != nil {
		Close(p[0])
		Close(p[1])
		return e
	}

	Close(p[1])
	p[1] = fd
	return nil
}

type envItem struct {
	name   *byte
	value  *byte
	nvalue int
}

type ProcAttr struct {
	Dir   string    // Current working directory.
	Env   []string  // Environment.
	Files []uintptr // File descriptors.
	Sys   *SysProcAttr
}

type SysProcAttr struct {
	Rfork int // additional flags to pass to rfork
}

var zeroProcAttr ProcAttr
var zeroSysProcAttr SysProcAttr

func forkExec(argv0 string, argv []string, attr *ProcAttr) (pid int, err error) {
	var (
		p      [2]int
		n      int
		errbuf [ERRMAX]byte
		wmsg   Waitmsg
	)

	if attr == nil {
		attr = &zeroProcAttr
	}
	sys := attr.Sys
	if sys == nil {
		sys = &zeroSysProcAttr
	}

	p[0] = -1
	p[1] = -1

	// Convert args to C form.
	argv0p, err := BytePtrFromString(argv0)
	if err != nil {
		return 0, err
	}
	argvp, err := SlicePtrFromStrings(argv)
	if err != nil {
		return 0, err
	}

	destDir := attr.Dir
	if destDir == "" {
		wdmu.Lock()
		destDir = wdStr
		wdmu.Unlock()
	}
	var dir *byte
	if destDir != "" {
		dir, err = BytePtrFromString(destDir)
		if err != nil {
			return 0, err
		}
	}
	var envvParsed []envItem
	if attr.Env != nil {
		envvParsed = make([]envItem, 0, len(attr.Env))
		for _, v := range attr.Env {
			i := 0
			for i < len(v) && v[i] != '=' {
				i++
			}

			envname, err := BytePtrFromString("/env/" + v[:i])
			if err != nil {
				return 0, err
			}
			envvalue := make([]byte, len(v)-i)
			copy(envvalue, v[i+1:])
			envvParsed = append(envvParsed, envItem{envname, &envvalue[0], len(v) - i})
		}
	}

	// Allocate child status pipe close on exec.
	e := cexecPipe(p[:])

	if e != nil {
		return 0, e
	}

	// Kick off child.
	pid, err = forkAndExecInChild(argv0p, argvp, envvParsed, dir, attr, p[1], sys.Rfork)

	if err != nil {
		if p[0] >= 0 {
			Close(p[0])
			Close(p[1])
		}
		return 0, err
	}

	// Read child error status from pipe.
	Close(p[1])
	n, err = Read(p[0], errbuf[:])
	Close(p[0])

	if err != nil || n != 0 {
		if n > 0 {
			err = NewError(string(errbuf[:n]))
		} else if err == nil {
			err = NewError("failed to read exec status")
		}

		// Child failed; wait for it to exit, to make sure
		// the zombies don't accumulate.
		for wmsg.Pid != pid {
			Await(&wmsg)
		}
		return 0, err
	}

	// Read got EOF, so pipe closed on exec, so exec succeeded.
	return pid, nil
}

type waitErr struct {
	Waitmsg
	err error
}

var procs struct {
	sync.Mutex
	waits map[int]chan *waitErr
}

// startProcess starts a new goroutine, tied to the OS
// thread, which runs the process and subsequently waits
// for it to finish, communicating the process stats back
// to any goroutines that may have been waiting on it.
//
// Such a dedicated goroutine is needed because on
// Plan 9, only the parent thread can wait for a child,
// whereas goroutines tend to jump OS threads (e.g.,
// between starting a process and running Wait(), the
// goroutine may have been rescheduled).
func startProcess(argv0 string, argv []string, attr *ProcAttr) (pid int, err error) {
	type forkRet struct {
		pid int
		err error
	}

	forkc := make(chan forkRet, 1)
	go func() {
		runtime.LockOSThread()
		var ret forkRet

		ret.pid, ret.err = forkExec(argv0, argv, attr)
		// If fork fails there is nothing to wait for.
		if ret.err != nil || ret.pid == 0 {
			forkc <- ret
			return
		}

		waitc := make(chan *waitErr, 1)

		// Mark that the process is running.
		procs.Lock()
		if procs.waits == nil {
			procs.waits = make(map[int]chan *waitErr)
		}
		procs.waits[ret.pid] = waitc
		procs.Unlock()

		forkc <- ret

		var w waitErr
		for w.err == nil && w.Pid != ret.pid {
			w.err = Await(&w.Waitmsg)
		}
		waitc <- &w
		close(waitc)
	}()
	ret := <-forkc
	return ret.pid, ret.err
}

// Combination of fork and exec, careful to be thread safe.
func ForkExec(argv0 string, argv []string, attr *ProcAttr) (pid int, err error) {
	return startProcess(argv0, argv, attr)
}

// StartProcess wraps ForkExec for package os.
func StartProcess(argv0 string, argv []string, attr *ProcAttr) (pid int, handle uintptr, err error) {
	pid, err = startProcess(argv0, argv, attr)
	return pid, 0, err
}

// Ordinary exec.
func Exec(argv0 string, argv []string, envv []string) (err error) {
	if envv != nil {
		r1, _, _ := RawSyscall(SYS_RFORK, RFCENVG, 0, 0)
		if int32(r1) == -1 {
			return NewError(errstr())
		}

		for _, v := range envv {
			i := 0
			for i < len(v) && v[i] != '=' {
				i++
			}

			fd, e := Create("/env/"+v[:i], O_WRONLY, 0666)
			if e != nil {
				return e
			}

			_, e = Write(fd, []byte(v[i+1:]))
			if e != nil {
				Close(fd)
				return e
			}
			Close(fd)
		}
	}

	argv0p, err := BytePtrFromString(argv0)
	if err != nil {
		return err
	}
	argvp, err := SlicePtrFromStrings(argv)
	if err != nil {
		return err
	}
	_, _, e1 := Syscall(SYS_EXEC,
		uintptr(unsafe.Pointer(argv0p)),
		uintptr(unsafe.Pointer(&argvp[0])),
		0)

	return e1
}

// WaitProcess waits until the pid of a
// running process is found in the queue of
// wait messages. It is used in conjunction
// with ForkExec/StartProcess to wait for a
// running process to exit.
func WaitProcess(pid int, w *Waitmsg) (err error) {
	procs.Lock()
	ch := procs.waits[pid]
	procs.Unlock()

	var wmsg *waitErr
	if ch != nil {
		wmsg = <-ch
		procs.Lock()
		if procs.waits[pid] == ch {
			delete(procs.waits, pid)
		}
		procs.Unlock()
	}
	if wmsg == nil {
		// ch was missing or ch is closed
		return NewError("process not found")
	}
	if wmsg.err != nil {
		return wmsg.err
	}
	if w != nil {
		*w = wmsg.Waitmsg
	}
	return nil
}
