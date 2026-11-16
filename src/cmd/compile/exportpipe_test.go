// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package main

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"golang.org/x/sys/unix"
)

func TestExportPipe(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "p.go")
	if err := os.WriteFile(src, []byte("package p\nfunc F() int { return 42 }\n"), 0666); err != nil {
		t.Fatal(err)
	}
	var (
		exportObj = filepath.Join(dir, "p.x")
		linkObj   = filepath.Join(dir, "p.o")
		fifo      = filepath.Join(dir, "exportpipe")
	)

	t.Run("RequiresLinkObj", func(t *testing.T) {
		out, err := runTestCompiler(t, "-p", "p", "-o", exportObj, "-exportpipe="+fifo, src)
		if err == nil {
			t.Fatalf("expected error without -linkobj, got success:\n%s", out)
		}
		if !strings.Contains(string(out), "-exportpipe requires -linkobj") {
			t.Fatalf("unexpected error output:\n%s", out)
		}
	})

	if err := unix.Mkfifo(fifo, 0600); err != nil {
		t.Fatalf("Mkfifo: %v", err)
	}

	t.Run("NoReaderDoesNotHang", func(t *testing.T) {
		out, err := runTestCompiler(t, "-p", "p", "-o", exportObj, "-linkobj", linkObj, "-exportpipe="+fifo, src)
		if err == nil {
			t.Fatalf("expected error opening readerless FIFO, got success:\n%s", out)
		}
		if !strings.Contains(string(out), "can't open export pipe") {
			t.Fatalf("unexpected error output:\n%s", out)
		}
	})

	t.Run("Notify", func(t *testing.T) {
		r0, err := os.OpenFile(fifo, os.O_RDONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			t.Fatalf("open r0: %v", err)
		}
		w, err := os.OpenFile(fifo, os.O_WRONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			r0.Close()
			t.Fatalf("open w: %v", err)
		}
		defer w.Close()
		r, err := os.Open(fifo)
		r0.Close()
		if err != nil {
			t.Fatalf("open r: %v", err)
		}
		defer r.Close()

		readDone := make(chan error, 1)
		go func() {
			var b [1]byte
			_, err := r.Read(b[:])
			readDone <- err
		}()

		out, err := runTestCompiler(t, "-p", "p", "-o", exportObj, "-linkobj", linkObj, "-exportpipe="+fifo, src)
		w.Close()
		if err != nil {
			t.Fatalf("compile failed: %v\n%s", err, out)
		}
		if err := <-readDone; err != nil {
			t.Fatalf("read from export pipe: %v", err)
		}
	})
}

func runTestCompiler(t *testing.T, args ...string) ([]byte, error) {
	t.Helper()
	testenv.MustHaveExec(t)
	if testCompiler == "" {
		t.Skip("testCompiler not available")
	}
	cmd := testenv.Command(t, testCompiler, args...)
	cmd.Env = append(cmd.Environ(), "COMPILE_TEST_EXEC_COMPILE=1")
	return cmd.CombinedOutput()
}
