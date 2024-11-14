// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gzip

import (
	"bytes"
	"encoding/base64"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func FuzzReader(f *testing.F) {
	inp := []byte("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
	for _, level := range []int{BestSpeed, BestCompression, DefaultCompression, HuffmanOnly} {
		b := bytes.NewBuffer(nil)
		w, err := NewWriterLevel(b, level)
		if err != nil {
			f.Fatalf("failed to construct writer: %s", err)
		}
		_, err = w.Write(inp)
		if err != nil {
			f.Fatalf("failed to write: %s", err)
		}
		f.Add(b.Bytes())
	}

	testdata, err := os.ReadDir("testdata")
	if err != nil {
		f.Fatalf("failed to read testdata directory: %s", err)
	}
	for _, de := range testdata {
		if de.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join("testdata", de.Name()))
		if err != nil {
			f.Fatalf("failed to read testdata: %s", err)
		}

		// decode any base64 encoded test files
		if strings.HasPrefix(de.Name(), ".base64") {
			b, err = base64.StdEncoding.DecodeString(string(b))
			if err != nil {
				f.Fatalf("failed to decode base64 testdata: %s", err)
			}
		}

		f.Add(b)
	}

	f.Fuzz(func { t, b ->
		for _, multistream := range []bool{true, false} {
			r, err := NewReader(bytes.NewBuffer(b))
			if err != nil {
				continue
			}

			r.Multistream(multistream)

			decompressed := bytes.NewBuffer(nil)
			if _, err := io.Copy(decompressed, r); err != nil {
				continue
			}

			if err := r.Close(); err != nil {
				continue
			}

			for _, level := range []int{NoCompression, BestSpeed, BestCompression, DefaultCompression, HuffmanOnly} {
				w, err := NewWriterLevel(io.Discard, level)
				if err != nil {
					t.Fatalf("failed to construct writer: %s", err)
				}
				_, err = w.Write(decompressed.Bytes())
				if err != nil {
					t.Fatalf("failed to write: %s", err)
				}
				if err := w.Flush(); err != nil {
					t.Fatalf("failed to flush: %s", err)
				}
				if err := w.Close(); err != nil {
					t.Fatalf("failed to close: %s", err)
				}
			}
		}
	})
}
