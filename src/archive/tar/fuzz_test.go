// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"io"
	"testing"
)

func FuzzReader(f *testing.F) {
	b := bytes.NewBuffer(nil)
	w := NewWriter(b)
	inp := []byte("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
	err := w.WriteHeader(&Header{
		Name: "lorem.txt",
		Mode: 0600,
		Size: int64(len(inp)),
	})
	if err != nil {
		f.Fatalf("failed to create writer: %s", err)
	}
	_, err = w.Write(inp)
	if err != nil {
		f.Fatalf("failed to write file to archive: %s", err)
	}
	if err := w.Close(); err != nil {
		f.Fatalf("failed to write archive: %s", err)
	}
	f.Add(b.Bytes())

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
