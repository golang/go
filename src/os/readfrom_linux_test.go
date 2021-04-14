// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"bytes"
	"internal/poll"
	"io"
	"math/rand"
	"os"
	. "os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"
)

func TestCopyFileRange(t *testing.T) {
	sizes := []int{
		1,
		42,
		1025,
		syscall.Getpagesize() + 1,
		32769,
	}
	t.Run("Basic", func(t *testing.T) {
		for _, size := range sizes {
			t.Run(strconv.Itoa(size), func(t *testing.T) {
				testCopyFileRange(t, int64(size), -1)
			})
		}
	})
	t.Run("Limited", func(t *testing.T) {
		t.Run("OneLess", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testCopyFileRange(t, int64(size), int64(size)-1)
				})
			}
		})
		t.Run("Half", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testCopyFileRange(t, int64(size), int64(size)/2)
				})
			}
		})
		t.Run("More", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testCopyFileRange(t, int64(size), int64(size)+7)
				})
			}
		})
	})
	t.Run("DoesntTryInAppendMode", func(t *testing.T) {
		dst, src, data, hook := newCopyFileRangeTest(t, 42)

		dst2, err := OpenFile(dst.Name(), O_RDWR|O_APPEND, 0755)
		if err != nil {
			t.Fatal(err)
		}
		defer dst2.Close()

		if _, err := io.Copy(dst2, src); err != nil {
			t.Fatal(err)
		}
		if hook.called {
			t.Fatal("called poll.CopyFileRange for destination in O_APPEND mode")
		}
		mustSeekStart(t, dst2)
		mustContainData(t, dst2, data) // through traditional means
	})
	t.Run("NotRegular", func(t *testing.T) {
		t.Run("BothPipes", func(t *testing.T) {
			hook := hookCopyFileRange(t)

			pr1, pw1, err := Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer pr1.Close()
			defer pw1.Close()

			pr2, pw2, err := Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer pr2.Close()
			defer pw2.Close()

			// The pipe is empty, and PIPE_BUF is large enough
			// for this, by (POSIX) definition, so there is no
			// need for an additional goroutine.
			data := []byte("hello")
			if _, err := pw1.Write(data); err != nil {
				t.Fatal(err)
			}
			pw1.Close()

			n, err := io.Copy(pw2, pr1)
			if err != nil {
				t.Fatal(err)
			}
			if n != int64(len(data)) {
				t.Fatalf("transfered %d, want %d", n, len(data))
			}
			if !hook.called {
				t.Fatalf("should have called poll.CopyFileRange")
			}
			pw2.Close()
			mustContainData(t, pr2, data)
		})
		t.Run("DstPipe", func(t *testing.T) {
			dst, src, data, hook := newCopyFileRangeTest(t, 255)
			dst.Close()

			pr, pw, err := Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer pr.Close()
			defer pw.Close()

			n, err := io.Copy(pw, src)
			if err != nil {
				t.Fatal(err)
			}
			if n != int64(len(data)) {
				t.Fatalf("transfered %d, want %d", n, len(data))
			}
			if !hook.called {
				t.Fatalf("should have called poll.CopyFileRange")
			}
			pw.Close()
			mustContainData(t, pr, data)
		})
		t.Run("SrcPipe", func(t *testing.T) {
			dst, src, data, hook := newCopyFileRangeTest(t, 255)
			src.Close()

			pr, pw, err := Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer pr.Close()
			defer pw.Close()

			// The pipe is empty, and PIPE_BUF is large enough
			// for this, by (POSIX) definition, so there is no
			// need for an additional goroutine.
			if _, err := pw.Write(data); err != nil {
				t.Fatal(err)
			}
			pw.Close()

			n, err := io.Copy(dst, pr)
			if err != nil {
				t.Fatal(err)
			}
			if n != int64(len(data)) {
				t.Fatalf("transfered %d, want %d", n, len(data))
			}
			if !hook.called {
				t.Fatalf("should have called poll.CopyFileRange")
			}
			mustSeekStart(t, dst)
			mustContainData(t, dst, data)
		})
	})
	t.Run("Nil", func(t *testing.T) {
		var nilFile *File
		anyFile, err := os.CreateTemp("", "")
		if err != nil {
			t.Fatal(err)
		}
		defer Remove(anyFile.Name())
		defer anyFile.Close()

		if _, err := io.Copy(nilFile, nilFile); err != ErrInvalid {
			t.Errorf("io.Copy(nilFile, nilFile) = %v, want %v", err, ErrInvalid)
		}
		if _, err := io.Copy(anyFile, nilFile); err != ErrInvalid {
			t.Errorf("io.Copy(anyFile, nilFile) = %v, want %v", err, ErrInvalid)
		}
		if _, err := io.Copy(nilFile, anyFile); err != ErrInvalid {
			t.Errorf("io.Copy(nilFile, anyFile) = %v, want %v", err, ErrInvalid)
		}

		if _, err := nilFile.ReadFrom(nilFile); err != ErrInvalid {
			t.Errorf("nilFile.ReadFrom(nilFile) = %v, want %v", err, ErrInvalid)
		}
		if _, err := anyFile.ReadFrom(nilFile); err != ErrInvalid {
			t.Errorf("anyFile.ReadFrom(nilFile) = %v, want %v", err, ErrInvalid)
		}
		if _, err := nilFile.ReadFrom(anyFile); err != ErrInvalid {
			t.Errorf("nilFile.ReadFrom(anyFile) = %v, want %v", err, ErrInvalid)
		}
	})
}

func testCopyFileRange(t *testing.T, size int64, limit int64) {
	dst, src, data, hook := newCopyFileRangeTest(t, size)

	// If we have a limit, wrap the reader.
	var (
		realsrc io.Reader
		lr      *io.LimitedReader
	)
	if limit >= 0 {
		lr = &io.LimitedReader{N: limit, R: src}
		realsrc = lr
		if limit < int64(len(data)) {
			data = data[:limit]
		}
	} else {
		realsrc = src
	}

	// Now call ReadFrom (through io.Copy), which will hopefully call
	// poll.CopyFileRange.
	n, err := io.Copy(dst, realsrc)
	if err != nil {
		t.Fatal(err)
	}

	// If we didn't have a limit, we should have called poll.CopyFileRange
	// with the right file descriptor arguments.
	if limit > 0 && !hook.called {
		t.Fatal("never called poll.CopyFileRange")
	}
	if hook.called && hook.dstfd != int(dst.Fd()) {
		t.Fatalf("wrong destination file descriptor: got %d, want %d", hook.dstfd, dst.Fd())
	}
	if hook.called && hook.srcfd != int(src.Fd()) {
		t.Fatalf("wrong source file descriptor: got %d, want %d", hook.srcfd, src.Fd())
	}

	// Check that the offsets after the transfer make sense, that the size
	// of the transfer was reported correctly, and that the destination
	// file contains exactly the bytes we expect it to contain.
	dstoff, err := dst.Seek(0, io.SeekCurrent)
	if err != nil {
		t.Fatal(err)
	}
	srcoff, err := src.Seek(0, io.SeekCurrent)
	if err != nil {
		t.Fatal(err)
	}
	if dstoff != srcoff {
		t.Errorf("offsets differ: dstoff = %d, srcoff = %d", dstoff, srcoff)
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

// newCopyFileRangeTest initializes a new test for copy_file_range.
//
// It creates source and destination files, and populates the source file
// with random data of the specified size. It also hooks package os' call
// to poll.CopyFileRange and returns the hook so it can be inspected.
func newCopyFileRangeTest(t *testing.T, size int64) (dst, src *File, data []byte, hook *copyFileRangeHook) {
	t.Helper()

	hook = hookCopyFileRange(t)
	tmp := t.TempDir()

	src, err := Create(filepath.Join(tmp, "src"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { src.Close() })

	dst, err = Create(filepath.Join(tmp, "dst"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dst.Close() })

	// Populate the source file with data, then rewind it, so it can be
	// consumed by copy_file_range(2).
	prng := rand.New(rand.NewSource(time.Now().Unix()))
	data = make([]byte, size)
	prng.Read(data)
	if _, err := src.Write(data); err != nil {
		t.Fatal(err)
	}
	if _, err := src.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}

	return dst, src, data, hook
}

// mustContainData ensures that the specified file contains exactly the
// specified data.
func mustContainData(t *testing.T, f *File, data []byte) {
	t.Helper()

	got := make([]byte, len(data))
	if _, err := io.ReadFull(f, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Fatalf("didn't get the same data back from %s", f.Name())
	}
	if _, err := f.Read(make([]byte, 1)); err != io.EOF {
		t.Fatalf("not at EOF")
	}
}

func mustSeekStart(t *testing.T, f *File) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
}

func hookCopyFileRange(t *testing.T) *copyFileRangeHook {
	h := new(copyFileRangeHook)
	h.install()
	t.Cleanup(h.uninstall)
	return h
}

type copyFileRangeHook struct {
	called bool
	dstfd  int
	srcfd  int
	remain int64

	original func(dst, src *poll.FD, remain int64) (int64, bool, error)
}

func (h *copyFileRangeHook) install() {
	h.original = *PollCopyFileRangeP
	*PollCopyFileRangeP = func(dst, src *poll.FD, remain int64) (int64, bool, error) {
		h.called = true
		h.dstfd = dst.Sysfd
		h.srcfd = src.Sysfd
		h.remain = remain
		return h.original(dst, src, remain)
	}
}

func (h *copyFileRangeHook) uninstall() {
	*PollCopyFileRangeP = h.original
}

// On some kernels copy_file_range fails on files in /proc.
func TestProcCopy(t *testing.T) {
	const cmdlineFile = "/proc/self/cmdline"
	cmdline, err := os.ReadFile(cmdlineFile)
	if err != nil {
		t.Skipf("can't read /proc file: %v", err)
	}
	in, err := os.Open(cmdlineFile)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	outFile := filepath.Join(t.TempDir(), "cmdline")
	out, err := os.Create(outFile)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(out, in); err != nil {
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	copy, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(cmdline, copy) {
		t.Errorf("copy of %q got %q want %q\n", cmdlineFile, copy, cmdline)
	}
}
