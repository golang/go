// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"fmt"
	"internal/poll"
	"os"
	"sync"
	"syscall"
	"testing"
)

type loggedFD struct {
	Net string
	FD  *poll.FD
	Err error
}

var (
	logMu     sync.Mutex
	loggedFDs map[syscall.Handle]*loggedFD
)

func logFD(net string, fd *poll.FD, err error) {
	logMu.Lock()
	defer logMu.Unlock()

	loggedFDs[fd.Sysfd] = &loggedFD{
		Net: net,
		FD:  fd,
		Err: err,
	}
}

func init() {
	loggedFDs = make(map[syscall.Handle]*loggedFD)
	*poll.LogInitFD = logFD
}

func findLoggedFD(h syscall.Handle) (lfd *loggedFD, found bool) {
	logMu.Lock()
	defer logMu.Unlock()

	lfd, found = loggedFDs[h]
	return lfd, found
}

// checkFileIsNotPartOfNetpoll verifies that f is not managed by netpoll.
// It returns error, if check fails.
func checkFileIsNotPartOfNetpoll(f *os.File) error {
	lfd, found := findLoggedFD(syscall.Handle(f.Fd()))
	if !found {
		return fmt.Errorf("%v fd=%v: is not found in the log", f.Name(), f.Fd())
	}
	if lfd.FD.IsPartOfNetpoll() {
		return fmt.Errorf("%v fd=%v: is part of netpoll, but should not be (logged: net=%v err=%v)", f.Name(), f.Fd(), lfd.Net, lfd.Err)
	}
	return nil
}

func TestFileFdsAreInitialised(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	err = checkFileIsNotPartOfNetpoll(f)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSerialFdsAreInitialised(t *testing.T) {
	for _, name := range []string{"COM1", "COM2", "COM3", "COM4"} {
		t.Run(name, func(t *testing.T) {
			h, err := syscall.CreateFile(syscall.StringToUTF16Ptr(name),
				syscall.GENERIC_READ|syscall.GENERIC_WRITE,
				0,
				nil,
				syscall.OPEN_EXISTING,
				syscall.FILE_ATTRIBUTE_NORMAL|syscall.FILE_FLAG_OVERLAPPED,
				0)
			if err != nil {
				if errno, ok := err.(syscall.Errno); ok {
					switch errno {
					case syscall.ERROR_FILE_NOT_FOUND,
						syscall.ERROR_ACCESS_DENIED:
						t.Log("Skipping: ", err)
						return
					}
				}
				t.Fatal(err)
			}
			f := os.NewFile(uintptr(h), name)
			defer f.Close()

			err = checkFileIsNotPartOfNetpoll(f)
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
