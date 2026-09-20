// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"internal/poll"
	"internal/syscall/windows"
	"io"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"testing"
	"time"
)

func newLazyFile(t testing.TB, overlapped bool) (*os.File, syscall.Handle) {
	t.Helper()
	name := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(name, []byte("hello"), 0600); err != nil {
		t.Fatal(err)
	}
	flags := syscall.O_RDWR
	if overlapped {
		flags |= windows.O_FILE_FLAG_OVERLAPPED
	}
	h := openHandle(t, name, flags)
	f := os.NewFile(uintptr(h), name)
	t.Cleanup(func() {
		if err := f.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Error(err)
		}
	})
	return f, h
}

func openHandle(t testing.TB, name string, flags int) syscall.Handle {
	t.Helper()
	h, err := syscall.Open(name, flags|syscall.O_CLOEXEC, 0600)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func TestNewFileLazyInit(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"sync", "overlapped"} {
		t.Run(mode, func(t *testing.T) {
			for _, first := range []string{"read", "seek", "deadline"} {
				t.Run(first, func(t *testing.T) {
					t.Parallel()
					f, _ := newLazyFile(t, mode == "overlapped")
					var buf [5]byte
					switch first {
					case "read":
						if n, err := f.Read(buf[:]); err != nil || n != len(buf) || string(buf[:]) != "hello" {
							t.Fatalf("Read = %q, %d, %v", buf, n, err)
						}
					case "seek":
						if off, err := f.Seek(1, io.SeekStart); err != nil || off != 1 {
							t.Fatalf("Seek = %d, %v", off, err)
						}
						if n, err := f.Read(buf[:1]); err != nil || n != 1 || buf[0] != 'e' {
							t.Fatalf("Read after Seek = %q, %d, %v", buf[:1], n, err)
						}
					case "deadline":
						err := f.SetReadDeadline(time.Now().Add(-time.Second))
						if mode == "sync" {
							if !errors.Is(err, os.ErrNoDeadline) {
								t.Fatalf("SetReadDeadline = %v; want ErrNoDeadline", err)
							}
						} else {
							if err != nil {
								t.Fatal(err)
							}
							if _, err := f.Read(buf[:1]); !errors.Is(err, os.ErrDeadlineExceeded) {
								t.Fatalf("Read = %v; want ErrDeadlineExceeded", err)
							}
						}
					}
				})
			}
		})
	}
}

func TestNewFileBlockedHandle(t *testing.T) {
	t.Parallel()
	for _, action := range []string{"close", "readClose", "deadlineControl"} {
		t.Run(action, func(t *testing.T) {
			t.Parallel()
			name := pipeName()
			writer := newBytePipe(t, name, true)
			h := openHandle(t, name, syscall.O_RDWR)
			var wg sync.WaitGroup
			wg.Go(func() {
				var buf [1]byte
				var n uint32
				syscall.ReadFile(h, buf[:], &n, nil)
			})
			time.Sleep(20 * time.Millisecond) // Let the native read block.
			f := os.NewFile(uintptr(h), name)
			defer func() {
				writer.Close()
				syscall.CancelIoEx(h, nil)
				wg.Wait()
				f.Close()
			}()

			if action == "readClose" {
				wg.Go(func() {
					var buf [1]byte
					_, err := f.Read(buf[:])
					if !errors.Is(err, os.ErrClosed) {
						t.Errorf("Read = %v; want ErrClosed", err)
					}
				})
				time.Sleep(20 * time.Millisecond) // Let first-use detection block.
			}
			if action == "deadlineControl" {
				// Unlike Read, a deadline setter does not hold the I/O locks.
				// Fd must also avoid waiting for the initialization lock.
				wg.Go(func() {
					if err := f.SetDeadline(time.Time{}); !errors.Is(err, os.ErrNoDeadline) {
						t.Errorf("SetDeadline = %v; want ErrNoDeadline", err)
					}
				})
				time.Sleep(20 * time.Millisecond) // Let first-use detection block.
			}
			if action == "close" || action == "readClose" {
				if err := f.Close(); err != nil {
					t.Fatal(err)
				}
				return
			}
			if got := f.Fd(); got != uintptr(h) {
				t.Errorf("Fd = %d; want %d", got, h)
			}
			raw, err := f.SyscallConn()
			if err == nil {
				err = raw.Control(func(fd uintptr) {
					if err := syscall.CancelIoEx(syscall.Handle(fd), nil); err != nil && err != syscall.ERROR_NOT_FOUND {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestNewFileEventFallback(t *testing.T) {
	t.Parallel()
	for _, fallback := range []string{"fd", "externalIOCP"} {
		t.Run(fallback, func(t *testing.T) {
			t.Parallel()
			f, h := newLazyFile(t, true)
			if fallback == "fd" {
				if got := f.Fd(); got != uintptr(h) {
					t.Fatalf("Fd = %d; want %d", got, h)
				}
			} else {
				iocp, err := windows.CreateIoCompletionPort(syscall.InvalidHandle, 0, 0, 0)
				if err != nil {
					t.Fatal(err)
				}
				defer syscall.CloseHandle(iocp)
				if _, err := windows.CreateIoCompletionPort(h, iocp, 0, 0); err != nil {
					t.Fatal(err)
				}
			}
			if err := f.SetDeadline(time.Time{}); !errors.Is(err, os.ErrNoDeadline) {
				t.Fatalf("SetDeadline = %v; want ErrNoDeadline", err)
			}
			var buf [5]byte
			if n, err := f.ReadAt(buf[:], 0); err != nil || n != len(buf) || string(buf[:]) != "hello" {
				t.Fatalf("ReadAt = %q, %d, %v", buf, n, err)
			}
		})
	}
}

func TestNewFileLazyInitRace(t *testing.T) {
	t.Parallel()
	for _, action := range []string{"close", "fd"} {
		t.Run(action, func(t *testing.T) {
			for range 100 {
				f, _ := newLazyFile(t, true)
				start := make(chan struct{})
				var wg sync.WaitGroup
				for _, op := range []func() error{
					func() error {
						var b [1]byte
						_, err := f.ReadAt(b[:], 0)
						return err
					},
					func() error { return f.SetDeadline(time.Time{}) },
				} {
					wg.Go(func() {
						<-start
						if err := op(); err != nil && !errors.Is(err, os.ErrClosed) && !errors.Is(err, os.ErrNoDeadline) && err != poll.ErrFileClosing {
							t.Error(err)
						}
					})
				}
				wg.Go(func() {
					<-start
					if action == "close" {
						if err := f.Close(); err != nil {
							t.Error(err)
						}
					} else {
						f.Fd()
					}
				})
				close(start)
				wg.Wait()
				f.Close()
			}
		})
	}
}
