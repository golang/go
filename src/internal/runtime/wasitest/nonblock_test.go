// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Not all systems have syscall.Mkfifo.
//go:build !aix && !plan9 && !solaris && !wasm && !windows

package wasi_test

import (
	"bufio"
	"fmt"
	"internal/testenv"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
)

// This test creates a set of FIFOs and writes to them in reverse order. It
// checks that the output order matches the write order. The test binary opens
// the FIFOs in their original order and spawns a goroutine for each that reads
// from the FIFO and writes the result to stderr. If I/O was blocking, all
// goroutines would be blocked waiting for one read call to return, and the
// output order wouldn't match.

type fifo struct {
	file *os.File
	path string
}

func TestNonblock(t *testing.T) {
	if target != "wasip1/wasm" {
		t.Skip()
	}

	switch os.Getenv("GOWASIRUNTIME") {
	case "wasmer":
		t.Skip("wasmer does not support non-blocking I/O")
	}

	testenv.MustHaveGoRun(t)

	for _, mode := range []string{"os.OpenFile", "os.NewFile"} {
		t.Run(mode, func(t *testing.T) {
			args := []string{"run", "./testdata/nonblock.go", mode}

			fifos := make([]*fifo, 8)
			for i := range fifos {
				path := filepath.Join(t.TempDir(), fmt.Sprintf("wasip1-nonblock-fifo-%d-%d", rand.Uint32(), i))
				if err := syscall.Mkfifo(path, 0666); err != nil {
					t.Fatal(err)
				}

				file, err := os.OpenFile(path, os.O_RDWR, 0)
				if err != nil {
					t.Fatal(err)
				}
				defer file.Close()

				args = append(args, path)
				fifos[len(fifos)-i-1] = &fifo{file, path}
			}

			subProcess := exec.Command(testenv.GoToolPath(t), args...)

			subProcess.Env = append(os.Environ(), "GOOS=wasip1", "GOARCH=wasm")

			pr, pw := io.Pipe()
			defer pw.Close()

			subProcess.Stderr = pw

			if err := subProcess.Start(); err != nil {
				t.Fatal(err)
			}

			scanner := bufio.NewScanner(pr)
			if !scanner.Scan() {
				t.Fatal("expected line:", scanner.Err())
			} else if scanner.Text() != "waiting" {
				t.Fatal("unexpected output:", scanner.Text())
			}

			for _, fifo := range fifos {
				if _, err := fifo.file.WriteString(fifo.path + "\n"); err != nil {
					t.Fatal(err)
				}
				if !scanner.Scan() {
					t.Fatal("expected line:", scanner.Err())
				} else if scanner.Text() != fifo.path {
					t.Fatal("unexpected line:", scanner.Text())
				}
			}

			if err := subProcess.Wait(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
