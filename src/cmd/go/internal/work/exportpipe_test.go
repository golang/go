// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestExportPipeNotify(t *testing.T) {
	if !earlyExportSupported {
		t.Skip("early export not supported on this platform")
	}

	objdir := t.TempDir() + string(filepath.Separator)
	ep, err := newExportPipe(objdir)
	if err != nil {
		t.Fatalf("newExportPipe: %v", err)
	}

	writeDone := make(chan error, 1)
	go func() {
		time.Sleep(10 * time.Millisecond)
		f, err := os.OpenFile(ep.name(), os.O_WRONLY, 0)
		if err != nil {
			writeDone <- err
			return
		}
		_, werr := f.Write([]byte{'\n'})
		cerr := f.Close()
		writeDone <- errors.Join(werr, cerr)
	}()

	if err := ep.wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if err := <-writeDone; err != nil {
		t.Fatalf("writer: %v", err)
	}

	ep.close()
	if runtime.GOOS != "windows" {
		if _, err := os.Stat(ep.name()); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("export pipe %s still exists after close: %v", ep.name(), err)
		}
	}
}

func TestExportPipeCloseUnblocksWait(t *testing.T) {
	if !earlyExportSupported {
		t.Skip("early export not supported on this platform")
	}

	objdir := t.TempDir() + string(filepath.Separator)
	ep, err := newExportPipe(objdir)
	if err != nil {
		t.Fatalf("newExportPipe: %v", err)
	}

	waitDone := make(chan error, 1)
	go func() {
		waitDone <- ep.wait()
	}()

	time.Sleep(10 * time.Millisecond)
	ep.close()

	select {
	case err := <-waitDone:
		if !errors.Is(err, io.EOF) {
			t.Fatalf("wait after close = %v, want io.EOF", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("wait did not unblock after close")
	}
}

func TestExportPipeCloseBeforeWait(t *testing.T) {
	if !earlyExportSupported {
		t.Skip("early export not supported on this platform")
	}

	objdir := t.TempDir() + string(filepath.Separator)
	ep, err := newExportPipe(objdir)
	if err != nil {
		t.Fatalf("newExportPipe: %v", err)
	}

	ep.close()
	if err := ep.wait(); !errors.Is(err, io.EOF) {
		t.Fatalf("wait after close = %v, want io.EOF", err)
	}
}

func TestExportPipeCloseWithoutWait(t *testing.T) {
	if !earlyExportSupported {
		t.Skip("early export not supported on this platform")
	}

	objdir := t.TempDir() + string(filepath.Separator)
	ep, err := newExportPipe(objdir)
	if err != nil {
		t.Fatalf("newExportPipe: %v", err)
	}

	ep.close()
	if runtime.GOOS != "windows" {
		if _, err := os.Stat(ep.name()); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("export pipe %s still exists after close: %v", ep.name(), err)
		}
	}
}
