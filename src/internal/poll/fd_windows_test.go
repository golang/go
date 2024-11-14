// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"errors"
	"fmt"
	"internal/poll"
	"internal/syscall/windows"
	"os"
	"sync"
	"syscall"
	"testing"
	"unsafe"
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

	poll.InitWSA()
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
		t.Run(name, func { t ->
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

func TestWSASocketConflict(t *testing.T) {
	s, err := windows.WSASocket(syscall.AF_INET, syscall.SOCK_STREAM, syscall.IPPROTO_TCP, nil, 0, windows.WSA_FLAG_OVERLAPPED)
	if err != nil {
		t.Fatal(err)
	}
	fd := poll.FD{Sysfd: s, IsStream: true, ZeroReadIsEOF: true}
	_, err = fd.Init("tcp", true)
	if err != nil {
		syscall.CloseHandle(s)
		t.Fatal(err)
	}
	defer fd.Close()

	const SIO_TCP_INFO = syscall.IOC_INOUT | syscall.IOC_VENDOR | 39
	inbuf := uint32(0)
	var outbuf _TCP_INFO_v0
	cbbr := uint32(0)

	var ov syscall.Overlapped
	// Create an event so that we can efficiently wait for completion
	// of a requested overlapped I/O operation.
	ov.HEvent, _ = windows.CreateEvent(nil, 0, 0, nil)
	if ov.HEvent == 0 {
		t.Fatalf("could not create the event!")
	}
	defer syscall.CloseHandle(ov.HEvent)

	if err = fd.WSAIoctl(
		SIO_TCP_INFO,
		(*byte)(unsafe.Pointer(&inbuf)),
		uint32(unsafe.Sizeof(inbuf)),
		(*byte)(unsafe.Pointer(&outbuf)),
		uint32(unsafe.Sizeof(outbuf)),
		&cbbr,
		&ov,
		0,
	); err != nil && !errors.Is(err, syscall.ERROR_IO_PENDING) {
		t.Fatalf("could not perform the WSAIoctl: %v", err)
	}

	if err != nil && errors.Is(err, syscall.ERROR_IO_PENDING) {
		// It is possible that the overlapped I/O operation completed
		// immediately so there is no need to wait for it to complete.
		if res, err := syscall.WaitForSingleObject(ov.HEvent, syscall.INFINITE); res != 0 {
			t.Fatalf("waiting for the completion of the overlapped IO failed: %v", err)
		}
	}
}

type _TCP_INFO_v0 struct {
	State             uint32
	Mss               uint32
	ConnectionTimeMs  uint64
	TimestampsEnabled bool
	RttUs             uint32
	MinRttUs          uint32
	BytesInFlight     uint32
	Cwnd              uint32
	SndWnd            uint32
	RcvWnd            uint32
	RcvBuf            uint32
	BytesOut          uint64
	BytesIn           uint64
	BytesReordered    uint32
	BytesRetrans      uint32
	FastRetrans       uint32
	DupAcksIn         uint32
	TimeoutEpisodes   uint32
	SynRetrans        uint8
}
