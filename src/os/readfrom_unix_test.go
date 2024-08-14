// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || solaris

package os_test

import (
	"bytes"
	"io"
	"math/rand"
	. "os"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

type (
	copyFileTestFunc func(*testing.T, int64) (*File, *File, []byte, *copyFileHook, string)
	copyFileTestHook func(*testing.T) (*copyFileHook, string)
)

func TestCopyFile(t *testing.T) {
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
				testCopyFiles(t, int64(size), -1)
			})
		}
	})
	t.Run("Limited", func(t *testing.T) {
		t.Run("OneLess", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testCopyFiles(t, int64(size), int64(size)-1)
				})
			}
		})
		t.Run("Half", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testCopyFiles(t, int64(size), int64(size)/2)
				})
			}
		})
		t.Run("More", func(t *testing.T) {
			for _, size := range sizes {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					testCopyFiles(t, int64(size), int64(size)+7)
				})
			}
		})
	})
	t.Run("DoesntTryInAppendMode", func(t *testing.T) {
		for _, newTest := range copyFileTests {
			dst, src, data, hook, testName := newTest(t, 42)

			dst2, err := OpenFile(dst.Name(), O_RDWR|O_APPEND, 0755)
			if err != nil {
				t.Fatalf("%s: %v", testName, err)
			}
			defer dst2.Close()

			if _, err := io.Copy(dst2, src); err != nil {
				t.Fatalf("%s: %v", testName, err)
			}
			switch runtime.GOOS {
			case "illumos", "solaris": // sendfile() on SunOS allows target file with O_APPEND set.
				if !hook.called {
					t.Fatalf("%s: should have called the hook even with destination in O_APPEND mode", testName)
				}
			default:
				if hook.called {
					t.Fatalf("%s: hook shouldn't be called with destination in O_APPEND mode", testName)
				}
			}
			mustSeekStart(t, dst2)
			mustContainData(t, dst2, data) // through traditional means
		}
	})
	t.Run("CopyFileItself", func(t *testing.T) {
		for _, hookFunc := range copyFileHooks {
			hook, testName := hookFunc(t)

			f, err := CreateTemp("", "file-readfrom-itself-test")
			if err != nil {
				t.Fatalf("%s: failed to create tmp file: %v", testName, err)
			}
			t.Cleanup(func() {
				f.Close()
				Remove(f.Name())
			})

			data := []byte("hello world!")
			if _, err := f.Write(data); err != nil {
				t.Fatalf("%s: failed to create and feed the file: %v", testName, err)
			}

			if err := f.Sync(); err != nil {
				t.Fatalf("%s: failed to save the file: %v", testName, err)
			}

			// Rewind it.
			if _, err := f.Seek(0, io.SeekStart); err != nil {
				t.Fatalf("%s: failed to rewind the file: %v", testName, err)
			}

			// Read data from the file itself.
			if _, err := io.Copy(f, f); err != nil {
				t.Fatalf("%s: failed to read from the file: %v", testName, err)
			}

			if hook.written != 0 || hook.handled || hook.err != nil {
				t.Fatalf("%s: File.readFrom is expected not to use any zero-copy techniques when copying itself."+
					"got hook.written=%d, hook.handled=%t, hook.err=%v; expected hook.written=0, hook.handled=false, hook.err=nil",
					testName, hook.written, hook.handled, hook.err)
			}

			switch testName {
			case "hookCopyFileRange":
				// For copy_file_range(2), it fails and returns EINVAL when the source and target
				// refer to the same file and their ranges overlap. The hook should be called to
				// get the returned error and fall back to generic copy.
				if !hook.called {
					t.Fatalf("%s: should have called the hook", testName)
				}
			case "hookSendFile", "hookSendFileOverCopyFileRange":
				// For sendfile(2), it allows the source and target to refer to the same file and overlap.
				// The hook should not be called and just fall back to generic copy directly.
				if hook.called {
					t.Fatalf("%s: shouldn't have called the hook", testName)
				}
			default:
				t.Fatalf("%s: unexpected test", testName)
			}

			// Rewind it.
			if _, err := f.Seek(0, io.SeekStart); err != nil {
				t.Fatalf("%s: failed to rewind the file: %v", testName, err)
			}

			data2, err := io.ReadAll(f)
			if err != nil {
				t.Fatalf("%s: failed to read from the file: %v", testName, err)
			}

			// It should wind up a double of the original data.
			if s := strings.Repeat(string(data), 2); s != string(data2) {
				t.Fatalf("%s: file contained %s, expected %s", testName, data2, s)
			}
		}
	})
	t.Run("NotRegular", func(t *testing.T) {
		t.Run("BothPipes", func(t *testing.T) {
			for _, hookFunc := range copyFileHooks {
				hook, testName := hookFunc(t)

				pr1, pw1, err := Pipe()
				if err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				defer pr1.Close()
				defer pw1.Close()

				pr2, pw2, err := Pipe()
				if err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				defer pr2.Close()
				defer pw2.Close()

				// The pipe is empty, and PIPE_BUF is large enough
				// for this, by (POSIX) definition, so there is no
				// need for an additional goroutine.
				data := []byte("hello")
				if _, err := pw1.Write(data); err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				pw1.Close()

				n, err := io.Copy(pw2, pr1)
				if err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				if n != int64(len(data)) {
					t.Fatalf("%s: transferred %d, want %d", testName, n, len(data))
				}
				switch runtime.GOOS {
				case "illumos", "solaris":
					// On SunOS, We rely on File.Stat to get the size of the source file,
					// which doesn't work for pipe.
					if hook.called {
						t.Fatalf("%s: shouldn't have called the hook with a source of pipe", testName)
					}
				default:
					if !hook.called {
						t.Fatalf("%s: should have called the hook with a source of pipe", testName)
					}
				}
				pw2.Close()
				mustContainData(t, pr2, data)
			}
		})
		t.Run("DstPipe", func(t *testing.T) {
			for _, newTest := range copyFileTests {
				dst, src, data, hook, testName := newTest(t, 255)
				dst.Close()

				pr, pw, err := Pipe()
				if err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				defer pr.Close()
				defer pw.Close()

				n, err := io.Copy(pw, src)
				if err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				if n != int64(len(data)) {
					t.Fatalf("%s: transferred %d, want %d", testName, n, len(data))
				}
				if !hook.called {
					t.Fatalf("%s: should have called the hook", testName)
				}
				pw.Close()
				mustContainData(t, pr, data)
			}
		})
		t.Run("SrcPipe", func(t *testing.T) {
			for _, newTest := range copyFileTests {
				dst, src, data, hook, testName := newTest(t, 255)
				src.Close()

				pr, pw, err := Pipe()
				if err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				defer pr.Close()
				defer pw.Close()

				// The pipe is empty, and PIPE_BUF is large enough
				// for this, by (POSIX) definition, so there is no
				// need for an additional goroutine.
				if _, err := pw.Write(data); err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				pw.Close()

				n, err := io.Copy(dst, pr)
				if err != nil {
					t.Fatalf("%s: %v", testName, err)
				}
				if n != int64(len(data)) {
					t.Fatalf("%s: transferred %d, want %d", testName, n, len(data))
				}
				switch runtime.GOOS {
				case "illumos", "solaris":
					// On SunOS, We rely on File.Stat to get the size of the source file,
					// which doesn't work for pipe.
					if hook.called {
						t.Fatalf("%s: shouldn't have called the hook with a source of pipe", testName)
					}
				default:
					if !hook.called {
						t.Fatalf("%s: should have called the hook with a source of pipe", testName)
					}
				}
				mustSeekStart(t, dst)
				mustContainData(t, dst, data)
			}
		})
	})
	t.Run("Nil", func(t *testing.T) {
		var nilFile *File
		anyFile, err := CreateTemp("", "")
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

func testCopyFile(t *testing.T, dst, src *File, data []byte, hook *copyFileHook, limit int64, testName string) {
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
	// poll.CopyFileRange or poll.SendFile.
	n, err := io.Copy(dst, realsrc)
	if err != nil {
		t.Fatalf("%s: %v", testName, err)
	}

	// If we didn't have a limit or had a positive limit, we should have called
	// poll.CopyFileRange or poll.SendFile with the right file descriptor arguments.
	if limit != 0 && !hook.called {
		t.Fatalf("%s: never called the hook", testName)
	}
	if hook.called && hook.dstfd != int(dst.Fd()) {
		t.Fatalf("%s: wrong destination file descriptor: got %d, want %d", testName, hook.dstfd, dst.Fd())
	}
	if hook.called && hook.srcfd != int(src.Fd()) {
		t.Fatalf("%s: wrong source file descriptor: got %d, want %d", testName, hook.srcfd, src.Fd())
	}

	// Check that the offsets after the transfer make sense, that the size
	// of the transfer was reported correctly, and that the destination
	// file contains exactly the bytes we expect it to contain.
	dstoff, err := dst.Seek(0, io.SeekCurrent)
	if err != nil {
		t.Fatalf("%s: %v", testName, err)
	}
	srcoff, err := src.Seek(0, io.SeekCurrent)
	if err != nil {
		t.Fatalf("%s: %v", testName, err)
	}
	if dstoff != srcoff {
		t.Errorf("%s: offsets differ: dstoff = %d, srcoff = %d", testName, dstoff, srcoff)
	}
	if dstoff != int64(len(data)) {
		t.Errorf("%s: dstoff = %d, want %d", testName, dstoff, len(data))
	}
	if n != int64(len(data)) {
		t.Errorf("%s: short ReadFrom: wrote %d bytes, want %d", testName, n, len(data))
	}
	mustSeekStart(t, dst)
	mustContainData(t, dst, data)

	// If we had a limit, check that it was updated.
	if lr != nil {
		if want := limit - n; lr.N != want {
			t.Fatalf("%s: didn't update limit correctly: got %d, want %d", testName, lr.N, want)
		}
	}
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

// newCopyFileTest initializes a new test for copying data between files.
// It creates source and destination files, and populates the source file
// with random data of the specified size, then rewind it, so it can be
// consumed by copy_file_range(2) or sendfile(2).
func newCopyFileTest(t *testing.T, size int64) (dst, src *File, data []byte) {
	src, data = createTempFile(t, "test-copy-file-src", size)

	dst, err := CreateTemp(t.TempDir(), "test-copy-file-dst")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dst.Close() })

	return
}

type copyFileHook struct {
	called bool
	dstfd  int
	srcfd  int

	written int64
	handled bool
	err     error
}

func createTempFile(t *testing.T, name string, size int64) (*File, []byte) {
	f, err := CreateTemp(t.TempDir(), name)
	if err != nil {
		t.Fatalf("failed to create temporary file: %v", err)
	}
	t.Cleanup(func() {
		f.Close()
	})

	randSeed := time.Now().Unix()
	t.Logf("random data seed: %d\n", randSeed)
	prng := rand.New(rand.NewSource(randSeed))
	data := make([]byte, size)
	prng.Read(data)
	if _, err := f.Write(data); err != nil {
		t.Fatalf("failed to create and feed the file: %v", err)
	}
	if err := f.Sync(); err != nil {
		t.Fatalf("failed to save the file: %v", err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("failed to rewind the file: %v", err)
	}

	return f, data
}
