// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"bytes"
	"errors"
	"internal/poll"
	"internal/testpty"
	"io"
	"math/rand"
	"net"
	. "os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"syscall"
	"testing"
	"time"

	"golang.org/x/net/nettest"
)

func TestSpliceFile(t *testing.T) {
	sizes := []int{
		1,
		42,
		1025,
		syscall.Getpagesize() + 1,
		32769,
	}
	t.Run("Basic-TCP", func(t *testing.T) {
		for _, size := range sizes {
			t.Run(strconv.Itoa(size), func(t *testing.T) {
				testSpliceFile(t, "tcp", int64(size), -1)
			})
		}
	})
	t.Run("Basic-Unix", func(t *testing.T) {
		for _, size := range sizes {
			t.Run(strconv.Itoa(size), func(t *testing.T) {
				testSpliceFile(t, "unix", int64(size), -1)
			})
		}
	})
	t.Run("TCP-To-TTY", func(t *testing.T) {
		testSpliceToTTY(t, "tcp", 32768)
	})
	t.Run("Unix-To-TTY", func(t *testing.T) {
		testSpliceToTTY(t, "unix", 32768)
	})
	t.Run("Limited", func(t *testing.T) {
		t.Run("OneLess-TCP", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testSpliceFile(t, "tcp", int64(size), int64(size)-1)
				})
			}
		})
		t.Run("OneLess-Unix", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testSpliceFile(t, "unix", int64(size), int64(size)-1)
				})
			}
		})
		t.Run("Half-TCP", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testSpliceFile(t, "tcp", int64(size), int64(size)/2)
				})
			}
		})
		t.Run("Half-Unix", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testSpliceFile(t, "unix", int64(size), int64(size)/2)
				})
			}
		})
		t.Run("More-TCP", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testSpliceFile(t, "tcp", int64(size), int64(size)+1)
				})
			}
		})
		t.Run("More-Unix", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testSpliceFile(t, "unix", int64(size), int64(size)+1)
				})
			}
		})
	})
}

func testSpliceFile(t *testing.T, proto string, size, limit int64) {
	dst, src, data, hook, cleanup := newSpliceFileTest(t, proto, size)
	defer cleanup()

	// If we have a limit, wrap the reader.
	var (
		r  io.Reader
		lr *io.LimitedReader
	)
	if limit >= 0 {
		lr = &io.LimitedReader{N: limit, R: src}
		r = lr
		if limit < int64(len(data)) {
			data = data[:limit]
		}
	} else {
		r = src
	}
	// Now call ReadFrom (through io.Copy), which will hopefully call poll.Splice
	n, err := io.Copy(dst, r)
	if err != nil {
		t.Fatal(err)
	}

	// We should have called poll.Splice with the right file descriptor arguments.
	if n > 0 && !hook.called {
		t.Fatal("expected to called poll.Splice")
	}
	if hook.called && hook.dstfd != int(dst.Fd()) {
		t.Fatalf("wrong destination file descriptor: got %d, want %d", hook.dstfd, dst.Fd())
	}
	sc, ok := src.(syscall.Conn)
	if !ok {
		t.Fatalf("server Conn is not a syscall.Conn")
	}
	rc, err := sc.SyscallConn()
	if err != nil {
		t.Fatalf("server Conn SyscallConn error: %v", err)
	}
	if err = rc.Control(func(fd uintptr) {
		if hook.called && hook.srcfd != int(fd) {
			t.Fatalf("wrong source file descriptor: got %d, want %d", hook.srcfd, int(fd))
		}
	}); err != nil {
		t.Fatalf("server Conn Control error: %v", err)
	}

	// Check that the offsets after the transfer make sense, that the size
	// of the transfer was reported correctly, and that the destination
	// file contains exactly the bytes we expect it to contain.
	dstoff, err := dst.Seek(0, io.SeekCurrent)
	if err != nil {
		t.Fatal(err)
	}
	if dstoff != int64(len(data)) {
		t.Errorf("dstoff = %d, want %d", dstoff, len(data))
	}
	if n != int64(len(data)) {
		t.Errorf("short ReadFrom: wrote %d bytes, want %d", n, len(data))
	}
	mustSeekStart(t, dst)
	mustContainData(t, dst, data)

	// If we had a limit, check that it was updated.
	if lr != nil {
		if want := limit - n; lr.N != want {
			t.Fatalf("didn't update limit correctly: got %d, want %d", lr.N, want)
		}
	}
}

// Issue #59041.
func testSpliceToTTY(t *testing.T, proto string, size int64) {
	var wg sync.WaitGroup

	// Call wg.Wait as the final deferred function,
	// because the goroutines may block until some of
	// the deferred Close calls.
	defer wg.Wait()

	pty, ttyName, err := testpty.Open()
	if err != nil {
		t.Skipf("skipping test because pty open failed: %v", err)
	}
	defer pty.Close()

	// Open the tty directly, rather than via OpenFile.
	// This bypasses the non-blocking support and is required
	// to recreate the problem in the issue (#59041).
	ttyFD, err := syscall.Open(ttyName, syscall.O_RDWR, 0)
	if err != nil {
		t.Skipf("skipping test because failed to open tty: %v", err)
	}
	defer syscall.Close(ttyFD)

	tty := NewFile(uintptr(ttyFD), "tty")
	defer tty.Close()

	client, server := createSocketPair(t, proto)

	data := bytes.Repeat([]byte{'a'}, int(size))

	wg.Add(1)
	go func() {
		defer wg.Done()
		// The problem (issue #59041) occurs when writing
		// a series of blocks of data. It does not occur
		// when all the data is written at once.
		for i := 0; i < len(data); i += 1024 {
			if _, err := client.Write(data[i : i+1024]); err != nil {
				// If we get here because the client was
				// closed, skip the error.
				if !errors.Is(err, net.ErrClosed) {
					t.Errorf("error writing to socket: %v", err)
				}
				return
			}
		}
		client.Close()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		buf := make([]byte, 32)
		for {
			if _, err := pty.Read(buf); err != nil {
				if err != io.EOF && !errors.Is(err, ErrClosed) {
					// An error here doesn't matter for
					// our test.
					t.Logf("error reading from pty: %v", err)
				}
				return
			}
		}
	}()

	// Close Client to wake up the writing goroutine if necessary.
	defer client.Close()

	_, err = io.Copy(tty, server)
	if err != nil {
		t.Fatal(err)
	}
}

var (
	copyFileTests = []copyFileTestFunc{newCopyFileRangeTest, newSendfileOverCopyFileRangeTest}
	copyFileHooks = []copyFileTestHook{hookCopyFileRange, hookSendFileOverCopyFileRange}
)

func testCopyFiles(t *testing.T, size, limit int64) {
	testCopyFileRange(t, size, limit)
	testSendfileOverCopyFileRange(t, size, limit)
}

func testCopyFileRange(t *testing.T, size int64, limit int64) {
	dst, src, data, hook, name := newCopyFileRangeTest(t, size)
	testCopyFile(t, dst, src, data, hook, limit, name)
}

func testSendfileOverCopyFileRange(t *testing.T, size int64, limit int64) {
	dst, src, data, hook, name := newSendfileOverCopyFileRangeTest(t, size)
	testCopyFile(t, dst, src, data, hook, limit, name)
}

// newCopyFileRangeTest initializes a new test for copy_file_range.
//
// It hooks package os' call to poll.CopyFileRange and returns the hook,
// so it can be inspected.
func newCopyFileRangeTest(t *testing.T, size int64) (dst, src *File, data []byte, hook *copyFileHook, name string) {
	t.Helper()

	name = "newCopyFileRangeTest"

	dst, src, data = newCopyFileTest(t, size)
	hook, _ = hookCopyFileRange(t)

	return
}

// newSendfileOverCopyFileRangeTest initializes a new test for sendfile over copy_file_range.
// It hooks package os' call to poll.SendFile and returns the hook,
// so it can be inspected.
func newSendfileOverCopyFileRangeTest(t *testing.T, size int64) (dst, src *File, data []byte, hook *copyFileHook, name string) {
	t.Helper()

	name = "newSendfileOverCopyFileRangeTest"

	dst, src, data = newCopyFileTest(t, size)
	hook, _ = hookSendFileOverCopyFileRange(t)

	return
}

// newSpliceFileTest initializes a new test for splice.
//
// It creates source sockets and destination file, and populates the source sockets
// with random data of the specified size. It also hooks package os' call
// to poll.Splice and returns the hook so it can be inspected.
func newSpliceFileTest(t *testing.T, proto string, size int64) (*File, net.Conn, []byte, *spliceFileHook, func()) {
	t.Helper()

	hook := hookSpliceFile(t)

	client, server := createSocketPair(t, proto)

	dst, err := CreateTemp(t.TempDir(), "dst-splice-file-test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dst.Close() })

	randSeed := time.Now().Unix()
	t.Logf("random data seed: %d\n", randSeed)
	prng := rand.New(rand.NewSource(randSeed))
	data := make([]byte, size)
	prng.Read(data)

	done := make(chan struct{})
	go func() {
		client.Write(data)
		client.Close()
		close(done)
	}()

	return dst, server, data, hook, func() { <-done }
}

func hookCopyFileRange(t *testing.T) (hook *copyFileHook, name string) {
	name = "hookCopyFileRange"

	hook = new(copyFileHook)
	orig := *PollCopyFileRangeP
	t.Cleanup(func() {
		*PollCopyFileRangeP = orig
	})
	*PollCopyFileRangeP = func(dst, src *poll.FD, remain int64) (int64, bool, error) {
		hook.called = true
		hook.dstfd = dst.Sysfd
		hook.srcfd = src.Sysfd
		hook.written, hook.handled, hook.err = orig(dst, src, remain)
		return hook.written, hook.handled, hook.err
	}
	return
}

func hookSendFileOverCopyFileRange(t *testing.T) (*copyFileHook, string) {
	return hookSendFileTB(t), "hookSendFileOverCopyFileRange"
}

func hookSendFileTB(tb testing.TB) *copyFileHook {
	// Disable poll.CopyFileRange to force the fallback to poll.SendFile.
	originalCopyFileRange := *PollCopyFileRangeP
	*PollCopyFileRangeP = func(dst, src *poll.FD, remain int64) (written int64, handled bool, err error) {
		return 0, false, nil
	}

	hook := new(copyFileHook)
	orig := poll.TestHookDidSendFile
	tb.Cleanup(func() {
		*PollCopyFileRangeP = originalCopyFileRange
		poll.TestHookDidSendFile = orig
	})
	poll.TestHookDidSendFile = func(dstFD *poll.FD, src int, written int64, err error, handled bool) {
		hook.called = true
		hook.dstfd = dstFD.Sysfd
		hook.srcfd = src
		hook.written = written
		hook.err = err
		hook.handled = handled
	}
	return hook
}

func hookSpliceFile(t *testing.T) *spliceFileHook {
	h := new(spliceFileHook)
	h.install()
	t.Cleanup(h.uninstall)
	return h
}

type spliceFileHook struct {
	called bool
	dstfd  int
	srcfd  int
	remain int64

	written int64
	handled bool
	err     error

	original func(dst, src *poll.FD, remain int64) (int64, bool, error)
}

func (h *spliceFileHook) install() {
	h.original = *PollSpliceFile
	*PollSpliceFile = func(dst, src *poll.FD, remain int64) (int64, bool, error) {
		h.called = true
		h.dstfd = dst.Sysfd
		h.srcfd = src.Sysfd
		h.remain = remain
		h.written, h.handled, h.err = h.original(dst, src, remain)
		return h.written, h.handled, h.err
	}
}

func (h *spliceFileHook) uninstall() {
	*PollSpliceFile = h.original
}

// On some kernels copy_file_range fails on files in /proc.
func TestProcCopy(t *testing.T) {
	t.Parallel()

	const cmdlineFile = "/proc/self/cmdline"
	cmdline, err := ReadFile(cmdlineFile)
	if err != nil {
		t.Skipf("can't read /proc file: %v", err)
	}
	in, err := Open(cmdlineFile)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	outFile := filepath.Join(t.TempDir(), "cmdline")
	out, err := Create(outFile)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(out, in); err != nil {
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	copy, err := ReadFile(outFile)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(cmdline, copy) {
		t.Errorf("copy of %q got %q want %q\n", cmdlineFile, copy, cmdline)
	}
}

func TestGetPollFDAndNetwork(t *testing.T) {
	t.Run("tcp4", func(t *testing.T) { testGetPollFDAndNetwork(t, "tcp4") })
	t.Run("unix", func(t *testing.T) { testGetPollFDAndNetwork(t, "unix") })
}

func testGetPollFDAndNetwork(t *testing.T, proto string) {
	_, server := createSocketPair(t, proto)
	sc, ok := server.(syscall.Conn)
	if !ok {
		t.Fatalf("server Conn is not a syscall.Conn")
	}
	rc, err := sc.SyscallConn()
	if err != nil {
		t.Fatalf("server SyscallConn error: %v", err)
	}
	if err = rc.Control(func(fd uintptr) {
		pfd, network := GetPollFDAndNetwork(server)
		if pfd == nil {
			t.Fatalf("GetPollFDAndNetwork didn't return poll.FD")
		}
		if string(network) != proto {
			t.Fatalf("GetPollFDAndNetwork returned wrong network, got: %s, want: %s", network, proto)
		}
		if pfd.Sysfd != int(fd) {
			t.Fatalf("GetPollFDAndNetwork returned wrong poll.FD, got: %d, want: %d", pfd.Sysfd, int(fd))
		}
		if !pfd.IsStream {
			t.Fatalf("expected IsStream to be true")
		}
		if err = pfd.Init(proto, true); err == nil {
			t.Fatalf("Init should have failed with the initialized poll.FD and return EEXIST error")
		}
	}); err != nil {
		t.Fatalf("server Control error: %v", err)
	}
}

func createSocketPair(t *testing.T, proto string) (client, server net.Conn) {
	t.Helper()
	if !nettest.TestableNetwork(proto) {
		t.Skipf("%s does not support %q", runtime.GOOS, proto)
	}

	ln, err := nettest.NewLocalListener(proto)
	if err != nil {
		t.Fatalf("NewLocalListener error: %v", err)
	}
	t.Cleanup(func() {
		if ln != nil {
			ln.Close()
		}
		if client != nil {
			client.Close()
		}
		if server != nil {
			server.Close()
		}
	})
	ch := make(chan struct{})
	go func() {
		var err error
		server, err = ln.Accept()
		if err != nil {
			t.Errorf("Accept new connection error: %v", err)
		}
		ch <- struct{}{}
	}()
	client, err = net.Dial(proto, ln.Addr().String())
	<-ch
	if err != nil {
		t.Fatalf("Dial new connection error: %v", err)
	}
	return client, server
}
