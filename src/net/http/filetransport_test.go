// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"io"
	"os"
	"path/filepath"
	"testing"
)

func checker(t *testing.T) func(string, error) {
	return func(call string, err error) {
		if err == nil {
			return
		}
		t.Fatalf("%s: %v", call, err)
	}
}

func TestFileTransport(t *testing.T) {
	check := checker(t)

	dname := t.TempDir()
	fname := filepath.Join(dname, "foo.txt")
	err := os.WriteFile(fname, []byte("Bar"), 0644)
	check("WriteFile", err)
	defer os.Remove(fname)

	tr := &Transport{}
	tr.RegisterProtocol("file", NewFileTransport(Dir(dname)))
	c := &Client{Transport: tr}

	fooURLs := []string{"file:///foo.txt", "file://../foo.txt"}
	for _, urlstr := range fooURLs {
		res, err := c.Get(urlstr)
		check("Get "+urlstr, err)
		if res.StatusCode != 200 {
			t.Errorf("for %s, StatusCode = %d, want 200", urlstr, res.StatusCode)
		}
		if res.ContentLength != -1 {
			t.Errorf("for %s, ContentLength = %d, want -1", urlstr, res.ContentLength)
		}
		if res.Body == nil {
			t.Fatalf("for %s, nil Body", urlstr)
		}
		slurp, err := io.ReadAll(res.Body)
		res.Body.Close()
		check("ReadAll "+urlstr, err)
		if string(slurp) != "Bar" {
			t.Errorf("for %s, got content %q, want %q", urlstr, string(slurp), "Bar")
		}
	}

	const badURL = "file://../no-exist.txt"
	res, err := c.Get(badURL)
	check("Get "+badURL, err)
	if res.StatusCode != 404 {
		t.Errorf("for %s, StatusCode = %d, want 404", badURL, res.StatusCode)
	}
	res.Body.Close()
}
