// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/modfetch/codehost"
)

func Unzip(dir, zipfile, prefix string, maxSize int64) error {
	if maxSize == 0 {
		maxSize = codehost.MaxZipFile
	}

	// Directory can exist, but must be empty.
	// except maybe
	files, _ := ioutil.ReadDir(dir)
	if len(files) > 0 {
		return fmt.Errorf("target directory %v exists and is not empty", dir)
	}
	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}

	f, err := os.Open(zipfile)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}

	z, err := zip.NewReader(f, info.Size())
	if err != nil {
		return fmt.Errorf("unzip %v: %s", zipfile, err)
	}

	// Check total size.
	var size int64
	for _, zf := range z.File {
		if !strings.HasPrefix(zf.Name, prefix) {
			return fmt.Errorf("unzip %v: unexpected file name %s", zipfile, zf.Name)
		}
		if strings.HasSuffix(zf.Name, "/") {
			continue
		}
		s := int64(zf.UncompressedSize64)
		if s < 0 || maxSize-size < s {
			return fmt.Errorf("unzip %v: content too large", zipfile)
		}
		size += s
	}

	// Unzip, enforcing sizes checked earlier.
	for _, zf := range z.File {
		if strings.HasSuffix(zf.Name, "/") {
			continue
		}
		dst := filepath.Join(dir, zf.Name[len(prefix):])
		if err := os.MkdirAll(filepath.Dir(dst), 0777); err != nil {
			return err
		}
		w, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0444)
		if err != nil {
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		r, err := zf.Open()
		if err != nil {
			r.Close()
			w.Close()
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		lr := &io.LimitedReader{R: r, N: int64(zf.UncompressedSize64) + 1}
		_, err = io.Copy(w, lr)
		r.Close()
		if err != nil {
			w.Close()
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		if err := w.Close(); err != nil {
			return fmt.Errorf("unzip %v: %v", zipfile, err)
		}
		if lr.N <= 0 {
			return fmt.Errorf("unzip %v: content too large", zipfile)
		}
	}

	return nil
}
