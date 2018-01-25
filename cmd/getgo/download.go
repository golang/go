// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	currentVersionURL = "https://golang.org/VERSION?m=text"
	downloadURLPrefix = "https://dl.google.com/go"
)

// downloadGoVersion downloads and upacks the specific go version to dest/go.
func downloadGoVersion(version, ops, arch, dest string) error {
	suffix := "tar.gz"
	if ops == "windows" {
		suffix = "zip"
	}
	uri := fmt.Sprintf("%s/%s.%s-%s.%s", downloadURLPrefix, version, ops, arch, suffix)

	verbosef("Downloading %s", uri)

	req, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		return err
	}
	req.Header.Add("User-Agent", fmt.Sprintf("golang.org-getgo/%s", version))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("Downloading Go from %s failed: %v", uri, err)
	}
	if resp.StatusCode > 299 {
		return fmt.Errorf("Downloading Go from %s failed with HTTP status %s", uri, resp.Status)
	}
	defer resp.Body.Close()

	tmpf, err := ioutil.TempFile("", "go")
	if err != nil {
		return err
	}
	defer os.Remove(tmpf.Name())

	h := sha256.New()

	w := io.MultiWriter(tmpf, h)
	if _, err := io.Copy(w, resp.Body); err != nil {
		return err
	}

	verbosef("Downloading SHA %s.sha256", uri)

	sresp, err := http.Get(uri + ".sha256")
	if err != nil {
		return fmt.Errorf("Downloading Go sha256 from %s.sha256 failed: %v", uri, err)
	}
	defer sresp.Body.Close()
	if sresp.StatusCode > 299 {
		return fmt.Errorf("Downloading Go sha256 from %s.sha256 failed with HTTP status %s", uri, sresp.Status)
	}

	shasum, err := ioutil.ReadAll(sresp.Body)
	if err != nil {
		return err
	}

	// Check the shasum.
	sum := fmt.Sprintf("%x", h.Sum(nil))
	if sum != string(shasum) {
		return fmt.Errorf("Shasum mismatch %s vs. %s", sum, string(shasum))
	}

	unpackFunc := unpackTar
	if ops == "windows" {
		unpackFunc = unpackZip
	}
	if err := unpackFunc(tmpf.Name(), dest); err != nil {
		return fmt.Errorf("Unpacking Go to %s failed: %v", dest, err)
	}
	return nil
}

func unpack(dest, name string, fi os.FileInfo, r io.Reader) error {
	if strings.HasPrefix(name, "go/") {
		name = name[len("go/"):]
	}

	path := filepath.Join(dest, name)
	if fi.IsDir() {
		return os.MkdirAll(path, fi.Mode())
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, fi.Mode())
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, r)
	return err
}

func unpackTar(src, dest string) error {
	r, err := os.Open(src)
	if err != nil {
		return err
	}
	defer r.Close()

	archive, err := gzip.NewReader(r)
	if err != nil {
		return err
	}
	defer archive.Close()

	tarReader := tar.NewReader(archive)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}

		if err := unpack(dest, header.Name, header.FileInfo(), tarReader); err != nil {
			return err
		}
	}

	return nil
}

func unpackZip(src, dest string) error {
	zr, err := zip.OpenReader(src)
	if err != nil {
		return err
	}

	for _, f := range zr.File {
		fr, err := f.Open()
		if err != nil {
			return err
		}
		if err := unpack(dest, f.Name, f.FileInfo(), fr); err != nil {
			return err
		}
		fr.Close()
	}

	return nil
}

func getLatestGoVersion() (string, error) {
	resp, err := http.Get(currentVersionURL)
	if err != nil {
		return "", fmt.Errorf("Getting current Go version failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode > 299 {
		b, _ := ioutil.ReadAll(io.LimitReader(resp.Body, 1024))
		return "", fmt.Errorf("Could not get current Go version: HTTP %d: %q", resp.StatusCode, b)
	}
	version, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(version)), nil
}
