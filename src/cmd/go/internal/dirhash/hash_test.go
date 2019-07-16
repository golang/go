// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dirhash

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func h(s string) string {
	return fmt.Sprintf("%x", sha256.Sum256([]byte(s)))
}

func htop(k string, s string) string {
	sum := sha256.Sum256([]byte(s))
	return k + ":" + base64.StdEncoding.EncodeToString(sum[:])
}

func TestHash1(t *testing.T) {
	files := []string{"xyz", "abc"}
	open := func(name string) (io.ReadCloser, error) {
		return ioutil.NopCloser(strings.NewReader("data for " + name)), nil
	}
	want := htop("h1", fmt.Sprintf("%s  %s\n%s  %s\n", h("data for abc"), "abc", h("data for xyz"), "xyz"))
	out, err := Hash1(files, open)
	if err != nil {
		t.Fatal(err)
	}
	if out != want {
		t.Errorf("Hash1(...) = %s, want %s", out, want)
	}

	_, err = Hash1([]string{"xyz", "a\nbc"}, open)
	if err == nil {
		t.Error("Hash1: expected error on newline in filenames")
	}
}

func TestHashDir(t *testing.T) {
	dir, err := ioutil.TempDir("", "dirhash-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	if err := ioutil.WriteFile(filepath.Join(dir, "xyz"), []byte("data for xyz"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, "abc"), []byte("data for abc"), 0666); err != nil {
		t.Fatal(err)
	}
	want := htop("h1", fmt.Sprintf("%s  %s\n%s  %s\n", h("data for abc"), "prefix/abc", h("data for xyz"), "prefix/xyz"))
	out, err := HashDir(dir, "prefix", Hash1)
	if err != nil {
		t.Fatalf("HashDir: %v", err)
	}
	if out != want {
		t.Errorf("HashDir(...) = %s, want %s", out, want)
	}
}

func TestHashZip(t *testing.T) {
	f, err := ioutil.TempFile("", "dirhash-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	z := zip.NewWriter(f)
	w, err := z.Create("prefix/xyz")
	if err != nil {
		t.Fatal(err)
	}
	w.Write([]byte("data for xyz"))
	w, err = z.Create("prefix/abc")
	if err != nil {
		t.Fatal(err)
	}
	w.Write([]byte("data for abc"))
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	want := htop("h1", fmt.Sprintf("%s  %s\n%s  %s\n", h("data for abc"), "prefix/abc", h("data for xyz"), "prefix/xyz"))
	out, err := HashZip(f.Name(), Hash1)
	if err != nil {
		t.Fatalf("HashDir: %v", err)
	}
	if out != want {
		t.Errorf("HashDir(...) = %s, want %s", out, want)
	}
}

func TestDirFiles(t *testing.T) {
	dir, err := ioutil.TempDir("", "dirfiles-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	if err := ioutil.WriteFile(filepath.Join(dir, "xyz"), []byte("data for xyz"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, "abc"), []byte("data for abc"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "subdir"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, "subdir", "xyz"), []byte("data for subdir xyz"), 0666); err != nil {
		t.Fatal(err)
	}
	prefix := "foo/bar@v2.3.4"
	out, err := DirFiles(dir, prefix)
	if err != nil {
		t.Fatalf("DirFiles: %v", err)
	}
	for _, file := range out {
		if !strings.HasPrefix(file, prefix) {
			t.Errorf("Dir file = %s, want prefix %s", file, prefix)
		}
	}
}
