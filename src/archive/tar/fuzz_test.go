// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !android
// +build !android

package tar

import (
	"bytes"
	"internal/testenv"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var constrainedBuilders = map[string]bool{
	"windows-386-2012":  true,
	"windows-386-2008":  true,
	"js-wasm":           true,
	"android-amd64-emu": true,
}

func FuzzReader(f *testing.F) {
	if constrainedBuilders[testenv.Builder()] {
		f.Skip("builder is memory constrained")
	}
	testdata, err := os.ReadDir("testdata")
	if err != nil {
		f.Fatalf("failed to read testdata directory: %s", err)
	}
	for _, de := range testdata {
		if de.IsDir() {
			continue
		}
		if strings.Contains(de.Name(), "big") {
			// skip large archives so we don't kill builders with restricted
			// memory
			continue
		}
		b, err := os.ReadFile(filepath.Join("testdata", de.Name()))
		if err != nil {
			f.Fatalf("failed to read testdata: %s", err)
		}
		f.Add(b)
	}

	f.Fuzz(func(t *testing.T, b []byte) {
		r := NewReader(bytes.NewReader(b))
		type file struct {
			header  *Header
			content []byte
		}
		files := []file{}
		for {
			hdr, err := r.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				return
			}
			buf := bytes.NewBuffer(nil)
			if _, err := io.Copy(buf, r); err != nil {
				continue
			}
			files = append(files, file{header: hdr, content: buf.Bytes()})
		}

		// If we were unable to read anything out of the archive don't
		// bother trying to roundtrip it.
		if len(files) == 0 {
			return
		}

		out := bytes.NewBuffer(nil)
		w := NewWriter(out)
		for _, f := range files {
			if err := w.WriteHeader(f.header); err != nil {
				t.Fatalf("unable to write previously parsed header: %s", err)
			}
			if _, err := w.Write(f.content); err != nil {
				t.Fatalf("unable to write previously parsed content: %s", err)
			}
		}
		if err := w.Close(); err != nil {
			t.Fatalf("Unable to write archive: %s", err)
		}

		// TODO: We may want to check if the archive roundtrips. This would require
		// taking into account addition of the two zero trailer blocks that Writer.Close
		// appends.
	})
}
