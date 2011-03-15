// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fork, exec, wait, etc.

package syscall

import (
	"sync"
	"utf16"
)

// Windows doesn't have a good concept of just Exec in the documented API.
// However, the kernel32 CreateProcess does a good job with
// ForkExec.

var ForkLock sync.RWMutex

// Joins an array of string with sep
// From the "strings" package.  Modified.
func stringJoin(a []string, sep string, escape escapeFunc) string {
	if len(a) == 0 {
		return ""
	}
	if len(a) == 1 {
		return a[0]
	}
	n := len(sep) * (len(a) - 1)
	for i := 0; i < len(a); i++ {
		a[i] = escape(a[i])
		n += len(a[i])
	}

	b := make([]byte, n)
	bp := 0
	for i := 0; i < len(a); i++ {
		s := a[i]
		for j := 0; j < len(s); j++ {
			b[bp] = s[j]
			bp++
		}
		if i+1 < len(a) {
			s = sep
			for j := 0; j < len(s); j++ {
				b[bp] = s[j]
				bp++
			}
		}
	}
	return string(b)
}

//Env block is a sequence of null terminated strings followed by a null.
//Last bytes are two unicode nulls, or four null bytes.
func createEnvBlock(envv []string) *uint16 {
	if len(envv) == 0 {
		return &utf16.Encode([]int("\x00\x00"))[0]
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

	return &utf16.Encode([]int(string(b)))[0]
}

type escapeFunc func(s string) string

//escapes quotes by " -> ""
//Also string -> "string"
func escapeAddQuotes(s string) string {
	//normal ascii char, one byte wide
	rune := byte('"')
	l := len(s)
	n := 0
	for i := 0; i < l; i++ {
		if s[i] == rune {
			n++
		}
	}
	qs := make([]byte, l+n+2)

	qs[0] = rune
	j := 1
	for i := 0; i < l; i++ {
		qs[i+j] = s[i]
		if s[i] == rune {
			j++
			qs[i+j] = rune
		}
	}
	qs[len(qs)-1] = rune
	return string(qs)
}


func CloseOnExec(fd int) {
	SetHandleInformation(int32(fd), HANDLE_FLAG_INHERIT, 0)
}

func SetNonblock(fd int, nonblocking bool) (errno int) {
	return 0
}

type ProcAttr struct {
	Dir   string
	Env   []string
	Files []int
}

var zeroAttributes ProcAttr

// TODO(kardia): Add trace
//The command and arguments are passed via the Command line parameter.
func StartProcess(argv0 string, argv []string, attr *ProcAttr) (pid, handle int, err int) {
	if attr == nil {
		attr = &zeroAttributes
	}
	if len(attr.Files) > 3 {
		return 0, 0, EWINDOWS
	}

	// CreateProcess will throw an error if the dir is not set to a valid dir
	// thus get the working dir if dir is empty.
	dir := attr.Dir
	if dir == "" {
		if wd, ok := Getwd(); ok == 0 {
			dir = wd
		}
	}
	fd := attr.Files

	startupInfo := new(StartupInfo)
	processInfo := new(ProcessInformation)

	GetStartupInfo(startupInfo)

	startupInfo.Flags = STARTF_USESTDHANDLES
	startupInfo.StdInput = 0
	startupInfo.StdOutput = 0
	startupInfo.StdErr = 0

	// Acquire the fork lock so that no other threads
	// create new fds that are not yet close-on-exec
	// before we fork.
	ForkLock.Lock()
	defer ForkLock.Unlock()

	var currentProc, _ = GetCurrentProcess()
	if len(fd) > 0 && fd[0] > 0 {
		err := DuplicateHandle(currentProc, int32(fd[0]), currentProc, &startupInfo.StdInput, 0, true, DUPLICATE_SAME_ACCESS)
		if err != 0 {
			return 0, 0, err
		}
		defer CloseHandle(int32(startupInfo.StdInput))
	}
	if len(fd) > 1 && fd[1] > 0 {
		err := DuplicateHandle(currentProc, int32(fd[1]), currentProc, &startupInfo.StdOutput, 0, true, DUPLICATE_SAME_ACCESS)
		if err != 0 {
			return 0, 0, err
		}
		defer CloseHandle(int32(startupInfo.StdOutput))
	}
	if len(fd) > 2 && fd[2] > 0 {
		err := DuplicateHandle(currentProc, int32(fd[2]), currentProc, &startupInfo.StdErr, 0, true, DUPLICATE_SAME_ACCESS)
		if err != 0 {
			return 0, 0, err
		}
		defer CloseHandle(int32(startupInfo.StdErr))
	}
	if len(argv) == 0 {
		argv = []string{""}
	}
	// argv0 must not be longer then 256 chars
	// but the entire cmd line can have up to 32k chars (msdn)
	err = CreateProcess(
		nil,
		StringToUTF16Ptr(escapeAddQuotes(argv0)+" "+stringJoin(argv[1:], " ", escapeAddQuotes)),
		nil,  //ptr to struct lpProcessAttributes
		nil,  //ptr to struct lpThreadAttributes
		true, //bInheritHandles
		CREATE_UNICODE_ENVIRONMENT, //Flags
		createEnvBlock(attr.Env),   //env block, NULL uses parent env
		StringToUTF16Ptr(dir),
		startupInfo,
		processInfo)

	if err == 0 {
		pid = int(processInfo.ProcessId)
		handle = int(processInfo.Process)
		CloseHandle(processInfo.Thread)
	}
	return
}

// Ordinary exec.
func Exec(argv0 string, argv []string, envv []string) (err int) {
	return EWINDOWS
}
