// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
	"time"
)

func TestClosedFileDeadline(t *testing.T) {
	t.Parallel()

	for _, kind := range []string{"file", "pipe"} {
		t.Run(kind, func(t *testing.T) {
			var files []*os.File
			if kind == "pipe" {
				if runtime.GOOS == "js" || runtime.GOOS == "wasip1" {
					t.Skip("pipes are not supported")
				}
				r, w, err := os.Pipe()
				if err != nil {
					t.Fatal(err)
				}
				files = []*os.File{r, w}
			} else {
				f, err := os.Create(filepath.Join(t.TempDir(), "file"))
				if err != nil {
					t.Fatal(err)
				}
				files = []*os.File{f}
			}
			for _, f := range files {
				if err := f.Close(); err != nil {
					t.Fatal(err)
				}
				for _, test := range []struct {
					name string
					set  func(time.Time) error
				}{
					{"SetDeadline", f.SetDeadline},
					{"SetReadDeadline", f.SetReadDeadline},
					{"SetWriteDeadline", f.SetWriteDeadline},
				} {
					t.Run(test.name, func(t *testing.T) {
						if err := test.set(time.Time{}); !errors.Is(err, os.ErrClosed) {
							t.Errorf("%s: got %v, want ErrClosed", f.Name(), err)
						} else if runtime.GOOS != "plan9" && err != os.ErrClosed {
							t.Errorf("%s: got %T, want unwrapped ErrClosed", f.Name(), err)
						}
					})
				}
			}
		})
	}
}

func TestClosedFileReaddir(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("directory reads use the raw handle on Windows")
	}
	t.Parallel()

	for _, n := range []int{-1, 0, 1} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			f, err := os.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			for _, test := range []struct {
				name string
				read func() error
			}{
				{"ReadDir", func() error { _, err := f.ReadDir(n); return err }},
				{"Readdir", func() error { _, err := f.Readdir(n); return err }},
				{"Readdirnames", func() error { _, err := f.Readdirnames(n); return err }},
			} {
				t.Run(test.name, func(t *testing.T) {
					err := test.read()
					if !errors.Is(err, os.ErrClosed) {
						t.Errorf("got %v, want ErrClosed", err)
					}
					if pe, ok := errors.AsType[*os.PathError](err); !ok || pe.Op == "" {
						t.Errorf("got %#v, want a PathError naming the operation", err)
					}
				})
			}
		})
	}
}
