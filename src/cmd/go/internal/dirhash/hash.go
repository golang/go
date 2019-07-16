// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dirhash defines hashes over directory trees.
package dirhash

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var DefaultHash = Hash1

type Hash func(files []string, open func(string) (io.ReadCloser, error)) (string, error)

func Hash1(files []string, open func(string) (io.ReadCloser, error)) (string, error) {
	h := sha256.New()
	files = append([]string(nil), files...)
	sort.Strings(files)
	for _, file := range files {
		if strings.Contains(file, "\n") {
			return "", errors.New("filenames with newlines are not supported")
		}
		r, err := open(file)
		if err != nil {
			return "", err
		}
		hf := sha256.New()
		_, err = io.Copy(hf, r)
		r.Close()
		if err != nil {
			return "", err
		}
		fmt.Fprintf(h, "%x  %s\n", hf.Sum(nil), file)
	}
	return "h1:" + base64.StdEncoding.EncodeToString(h.Sum(nil)), nil
}

func HashDir(dir, prefix string, hash Hash) (string, error) {
	files, err := DirFiles(dir, prefix)
	if err != nil {
		return "", err
	}
	osOpen := func(name string) (io.ReadCloser, error) {
		return os.Open(filepath.Join(dir, strings.TrimPrefix(name, prefix)))
	}
	return hash(files, osOpen)
}

func DirFiles(dir, prefix string) ([]string, error) {
	var files []string
	dir = filepath.Clean(dir)
	err := filepath.Walk(dir, func(file string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel := file
		if dir != "." {
			rel = file[len(dir)+1:]
		}
		f := filepath.Join(prefix, rel)
		files = append(files, filepath.ToSlash(f))
		return nil
	})
	if err != nil {
		return nil, err
	}
	return files, nil
}

func HashZip(zipfile string, hash Hash) (string, error) {
	z, err := zip.OpenReader(zipfile)
	if err != nil {
		return "", err
	}
	defer z.Close()
	var files []string
	zfiles := make(map[string]*zip.File)
	for _, file := range z.File {
		files = append(files, file.Name)
		zfiles[file.Name] = file
	}
	zipOpen := func(name string) (io.ReadCloser, error) {
		f := zfiles[name]
		if f == nil {
			return nil, fmt.Errorf("file %q not found in zip", name) // should never happen
		}
		return f.Open()
	}
	return hash(files, zipOpen)
}
