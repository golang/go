// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func serveInputError(t *testing.T, chunks ...string) error {
	t.Helper()
	scratch := t.TempDir()
	scriptPath := filepath.Join(scratch, "script.json")
	if err := os.WriteFile(scriptPath, []byte(`{}`), 0o666); err != nil {
		t.Fatal(err)
	}
	opts := Options{
		Gopls:    []string{os.Args[0], "-mode=stdio"},
		Env:      append(os.Environ(), fakeEnv+"="+scriptPath, fakeLogEnv+"="+filepath.Join(scratch, "gopls.log")),
		Compiler: newFakeCompiler(),
		Stderr:   new(syncBuffer),
	}
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	discard := make(chan struct{})
	go func() {
		defer close(discard)
		var tmp [32768]byte
		for {
			if _, err := outR.Read(tmp[:]); err != nil {
				return
			}
		}
	}()
	done := make(chan error, 1)
	go func() { done <- Serve(t.Context(), inR, outW, opts) }()
	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		for _, c := range chunks {
			if _, err := io.WriteString(inW, c); err != nil {
				return
			}
		}
		_ = inW.Close()
	}()
	var err error
	select {
	case err = <-done:
	case <-time.After(waitLimit):
		t.Fatal("Serve did not return after the framed input")
	}
	_ = inR.Close()
	<-writerDone
	_ = outW.Close()
	<-discard
	_ = outR.Close()
	return err
}

func TestFramingRejectsManyHeaderLines(t *testing.T) {
	var b strings.Builder
	for range 40 {
		b.WriteString("X-Pad: x\r\n")
	}
	b.WriteString("Content-Length: 2\r\n\r\n{}")
	if err := serveInputError(t, b.String()); err == nil {
		t.Fatal("Serve accepted a 41-line header block, want an error")
	}
}
func TestFramingRejectsWideHeaderLine(t *testing.T) {
	wide := "X-Pad: " + strings.Repeat("x", 9<<10) + "\r\n"
	bye := func(body string) string {
		return "Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n" + body
	}
	in := wide + bye(`{"jsonrpc":"2.0","id":1,"method":"shutdown","params":null}`) +
		bye(`{"jsonrpc":"2.0","method":"exit","params":{}}`)
	if err := serveInputError(t, in); err == nil {
		t.Fatal("Serve accepted a header line past the byte bound, want an error")
	}
}

func TestFramingRejectsBodyPastCap(t *testing.T) {
	if n, err := parseLength("17825793"); err == nil {
		t.Fatalf("parseLength accepted 17825793 = %d, want an error past the 16 MiB cap", n)
	}
	huge := strings.Repeat("x", 17<<20)
	frame := func(body string) string {
		return "Content-Length: " + strconv.Itoa(len(body)) + "\r\n\r\n" + body
	}
	in := frame(huge) +
		frame(`{"jsonrpc":"2.0","id":2,"method":"shutdown","params":null}`) +
		frame(`{"jsonrpc":"2.0","method":"exit","params":{}}`)
	if err := serveInputError(t, in); err == nil {
		t.Fatal("Serve accepted a 17 MiB frame, want an error")
	}
}
