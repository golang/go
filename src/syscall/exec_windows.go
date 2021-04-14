// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fork, exec, wait, etc.

package syscall

import (
	"sync"
	"unicode/utf16"
	"unsafe"
)

var ForkLock sync.RWMutex

// EscapeArg rewrites command line argument s as prescribed
// in https://msdn.microsoft.com/en-us/library/ms880421.
// This function returns "" (2 double quotes) if s is empty.
// Alternatively, these transformations are done:
// - every back slash (\) is doubled, but only if immediately
//   followed by double quote (");
// - every double quote (") is escaped by back slash (\);
// - finally, s is wrapped with double quotes (arg -> "arg"),
//   but only if there is space or tab inside s.
func EscapeArg(s string) string {
	if len(s) == 0 {
		return `""`
	}
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"', '\\', ' ', '\t':
			// Some escaping required.
			b := make([]byte, 0, len(s)+2)
			b = appendEscapeArg(b, s)
			return string(b)
		}
	}
	return s
}

// appendEscapeArg escapes the string s, as per escapeArg,
// appends the result to b, and returns the updated slice.
func appendEscapeArg(b []byte, s string) []byte {
	if len(s) == 0 {
		return append(b, `""`...)
	}

	needsBackslash := false
	hasSpace := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"', '\\':
			needsBackslash = true
		case ' ', '\t':
			hasSpace = true
		}
	}

	if !needsBackslash && !hasSpace {
		// No special handling required; normal case.
		return append(b, s...)
	}
	if !needsBackslash {
		// hasSpace is true, so we need to quote the string.
		b = append(b, '"')
		b = append(b, s...)
		return append(b, '"')
	}

	if hasSpace {
		b = append(b, '"')
	}
	slashes := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		default:
			slashes = 0
		case '\\':
			slashes++
		case '"':
			for ; slashes > 0; slashes-- {
				b = append(b, '\\')
			}
			b = append(b, '\\')
		}
		b = append(b, c)
	}
	if hasSpace {
		for ; slashes > 0; slashes-- {
			b = append(b, '\\')
		}
		b = append(b, '"')
	}

	return b
}

// makeCmdLine builds a command line out of args by escaping "special"
// characters and joining the arguments with spaces.
func makeCmdLine(args []string) string {
	var b []byte
	for _, v := range args {
		if len(b) > 0 {
			b = append(b, ' ')
		}
		b = appendEscapeArg(b, v)
	}
	return string(b)
}

// createEnvBlock converts an array of environment strings into
// the representation required by CreateProcess: a sequence of NUL
// terminated strings followed by a nil.
// Last bytes are two UCS-2 NULs, or four NUL bytes.
func createEnvBlock(envv []string) *uint16 {
	if len(envv) == 0 {
		return &utf16.Encode([]rune("\x00\x00"))[0]
	}
	length := 0
	for _, s := range envv {
		length += len(s) + 1
	}
	length += 1

	b := make([]byte, length)
	i := 0
	for _, s := range envv {
		l := len(s)
		copy(b[i:i+l], []byte(s))
		copy(b[i+l:i+l+1], []byte{0})
		i = i + l + 1
	}
	copy(b[i:i+1], []byte{0})

	return &utf16.Encode([]rune(string(b)))[0]
}

func CloseOnExec(fd Handle) {
	SetHandleInformation(Handle(fd), HANDLE_FLAG_INHERIT, 0)
}

func SetNonblock(fd Handle, nonblocking bool) (err error) {
	return nil
}

// FullPath retrieves the full path of the specified file.
func FullPath(name string) (path string, err error) {
	p, err := UTF16PtrFromString(name)
	if err != nil {
		return "", err
	}
	n := uint32(100)
	for {
		buf := make([]uint16, n)
		n, err = GetFullPathName(p, uint32(len(buf)), &buf[0], nil)
		if err != nil {
			return "", err
		}
		if n <= uint32(len(buf)) {
			return UTF16ToString(buf[:n]), nil
		}
	}
}

func isSlash(c uint8) bool {
	return c == '\\' || c == '/'
}

func normalizeDir(dir string) (name string, err error) {
	ndir, err := FullPath(dir)
	if err != nil {
		return "", err
	}
	if len(ndir) > 2 && isSlash(ndir[0]) && isSlash(ndir[1]) {
		// dir cannot have \\server\share\path form
		return "", EINVAL
	}
	return ndir, nil
}

func volToUpper(ch int) int {
	if 'a' <= ch && ch <= 'z' {
		ch += 'A' - 'a'
	}
	return ch
}

func joinExeDirAndFName(dir, p string) (name string, err error) {
	if len(p) == 0 {
		return "", EINVAL
	}
	if len(p) > 2 && isSlash(p[0]) && isSlash(p[1]) {
		// \\server\share\path form
		return p, nil
	}
	if len(p) > 1 && p[1] == ':' {
		// has drive letter
		if len(p) == 2 {
			return "", EINVAL
		}
		if isSlash(p[2]) {
			return p, nil
		} else {
			d, err := normalizeDir(dir)
			if err != nil {
				return "", err
			}
			if volToUpper(int(p[0])) == volToUpper(int(d[0])) {
				return FullPath(d + "\\" + p[2:])
			} else {
				return FullPath(p)
			}
		}
	} else {
		// no drive letter
		d, err := normalizeDir(dir)
		if err != nil {
			return "", err
		}
		if isSlash(p[0]) {
			return FullPath(d[:2] + p)
		} else {
			return FullPath(d + "\\" + p)
		}
	}
}

type ProcAttr struct {
	Dir   string
	Env   []string
	Files []uintptr
	Sys   *SysProcAttr
}

type SysProcAttr struct {
	HideWindow        bool
	CmdLine           string // used if non-empty, else the windows command line is built by escaping the arguments passed to StartProcess
	CreationFlags     uint32
	Token             Token               // if set, runs new process in the security context represented by the token
	ProcessAttributes *SecurityAttributes // if set, applies these security attributes as the descriptor for the new process
	ThreadAttributes  *SecurityAttributes // if set, applies these security attributes as the descriptor for the main thread of the new process
	NoInheritHandles  bool                // if set, each inheritable handle in the calling process is not inherited by the new process
}

var zeroProcAttr ProcAttr
var zeroSysProcAttr SysProcAttr

func StartProcess(argv0 string, argv []string, attr *ProcAttr) (pid int, handle uintptr, err error) {
	if len(argv0) == 0 {
		return 0, 0, EWINDOWS
	}
	if attr == nil {
		attr = &zeroProcAttr
	}
	sys := attr.Sys
	if sys == nil {
		sys = &zeroSysProcAttr
	}

	if len(attr.Files) > 3 {
		return 0, 0, EWINDOWS
	}
	if len(attr.Files) < 3 {
		return 0, 0, EINVAL
	}

	if len(attr.Dir) != 0 {
		// StartProcess assumes that argv0 is relative to attr.Dir,
		// because it implies Chdir(attr.Dir) before executing argv0.
		// Windows CreateProcess assumes the opposite: it looks for
		// argv0 relative to the current directory, and, only once the new
		// process is started, it does Chdir(attr.Dir). We are adjusting
		// for that difference here by making argv0 absolute.
		var err error
		argv0, err = joinExeDirAndFName(attr.Dir, argv0)
		if err != nil {
			return 0, 0, err
		}
	}
	argv0p, err := UTF16PtrFromString(argv0)
	if err != nil {
		return 0, 0, err
	}

	var cmdline string
	// Windows CreateProcess takes the command line as a single string:
	// use attr.CmdLine if set, else build the command line by escaping
	// and joining each argument with spaces
	if sys.CmdLine != "" {
		cmdline = sys.CmdLine
	} else {
		cmdline = makeCmdLine(argv)
	}

	var argvp *uint16
	if len(cmdline) != 0 {
		argvp, err = UTF16PtrFromString(cmdline)
		if err != nil {
			return 0, 0, err
		}
	}

	var dirp *uint16
	if len(attr.Dir) != 0 {
		dirp, err = UTF16PtrFromString(attr.Dir)
		if err != nil {
			return 0, 0, err
		}
	}

	// Acquire the fork lock so that no other threads
	// create new fds that are not yet close-on-exec
	// before we fork.
	ForkLock.Lock()
	defer ForkLock.Unlock()

	p, _ := GetCurrentProcess()
	fd := make([]Handle, len(attr.Files))
	for i := range attr.Files {
		if attr.Files[i] > 0 {
			err := DuplicateHandle(p, Handle(attr.Files[i]), p, &fd[i], 0, true, DUPLICATE_SAME_ACCESS)
			if err != nil {
				return 0, 0, err
			}
			defer CloseHandle(Handle(fd[i]))
		}
	}
	si := new(StartupInfo)
	si.Cb = uint32(unsafe.Sizeof(*si))
	si.Flags = STARTF_USESTDHANDLES
	if sys.HideWindow {
		si.Flags |= STARTF_USESHOWWINDOW
		si.ShowWindow = SW_HIDE
	}
	si.StdInput = fd[0]
	si.StdOutput = fd[1]
	si.StdErr = fd[2]

	pi := new(ProcessInformation)

	flags := sys.CreationFlags | CREATE_UNICODE_ENVIRONMENT
	if sys.Token != 0 {
		err = CreateProcessAsUser(sys.Token, argv0p, argvp, sys.ProcessAttributes, sys.ThreadAttributes, !sys.NoInheritHandles, flags, createEnvBlock(attr.Env), dirp, si, pi)
	} else {
		err = CreateProcess(argv0p, argvp, sys.ProcessAttributes, sys.ThreadAttributes, !sys.NoInheritHandles, flags, createEnvBlock(attr.Env), dirp, si, pi)
	}
	if err != nil {
		return 0, 0, err
	}
	defer CloseHandle(Handle(pi.Thread))

	return int(pi.ProcessId), uintptr(pi.Process), nil
}

func Exec(argv0 string, argv []string, envv []string) (err error) {
	return EWINDOWS
}
