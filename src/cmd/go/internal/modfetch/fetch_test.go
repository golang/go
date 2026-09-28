// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !wasip1

package modfetch

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/mod/module"
	"golang.org/x/mod/sumdb/dirhash"
	modzip "golang.org/x/mod/zip"
)

func TestUnzipVerify(t *testing.T) {
	var (
		ctx = t.Context()
		src = t.TempDir()
		mod = module.Version{Path: "example.com/unzip", Version: "v1.0.0"}
	)
	if err := os.WriteFile(filepath.Join(src, "go.mod"), []byte("module example.com/unzip\n"), 0o666); err != nil {
		t.Fatal(err)
	}

	zipfile := filepath.Join(t.TempDir(), "v1.0.0.zip")
	zf, err := os.Create(zipfile)
	if err != nil {
		t.Fatal(err)
	}
	if err := modzip.CreateFromDir(zf, mod, src); err != nil {
		t.Fatal(err)
	}
	if err := zf.Close(); err != nil {
		t.Fatal(err)
	}
	ziphash, err := dirhash.HashZip(zipfile, dirhash.DefaultHash)
	if err != nil {
		t.Fatal(err)
	}

	unzip := func(verify func() error) (string, error) {
		dir, err := NewFetcher().Unzip(ctx, mod, zipfile, ziphash, verify)
		if dir != "" {
			t.Cleanup(func() { RemoveAll(dir) })
		}
		return dir, err
	}

	var (
		errVerify      = errors.New("verify failed")
		verifyFnFails  = func() error { return errVerify }
		verifyFnPasses = func() error { return nil }
	)
	if _, err := unzip(verifyFnFails); !errors.Is(err, errVerify) {
		t.Fatalf("unzip with empty cache: got %v, want %v", err, errVerify)
	}
	if _, err := DownloadDir(ctx, mod); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("DownloadDir after failed verify: got %v, want %v", err, fs.ErrNotExist)
	}

	if _, err := unzip(verifyFnPasses); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadDir(ctx, mod); err != nil {
		t.Fatal(err)
	}
	if ok, err := haveZipHash(ctx, mod, ziphash); err != nil || !ok {
		t.Fatalf("haveZipHash = (%v, %v), want (true, nil)", ok, err)
	}

	if _, err := unzip(verifyFnFails); err != nil {
		t.Fatalf("unzip with cached copy: got %v, want nil", err)
	}

	if err := writeZipHash(ctx, mod, "h1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="); err != nil {
		t.Fatal(err)
	}
	if _, err := unzip(verifyFnFails); !errors.Is(err, errVerify) {
		t.Fatalf("unzip with mismatched ziphash: got %v, want %v", err, errVerify)
	}
	if _, err := DownloadDir(ctx, mod); err != nil {
		t.Fatal(err)
	}

	if _, err := unzip(verifyFnPasses); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadDir(ctx, mod); err != nil {
		t.Fatal(err)
	}

	if ok, err := haveZipHash(ctx, mod, ziphash); err != nil || !ok {
		t.Errorf("haveZipHash after re-unpack: got (%v, %v), want (true, nil)", ok, err)
	}
}
