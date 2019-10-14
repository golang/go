// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stdio_test

import (
	"io"
	"os"
	"path/filepath"
	"strings"
)

// overlayDir makes a minimal-overhead copy of srcRoot in which new files may be added.
//
// TODO: Once we no longer need to support the misc module in GOPATH mode,
// factor this function out into a package to reduce duplication.
func overlayDir(dstRoot, srcRoot string) error {
	dstRoot = filepath.Clean(dstRoot)
	if err := os.MkdirAll(dstRoot, 0777); err != nil {
		return err
	}

	symBase, err := filepath.Rel(srcRoot, dstRoot)
	if err != nil {
		symBase, err = filepath.Abs(srcRoot)
		if err != nil {
			return err
		}
	}

	return filepath.Walk(srcRoot, func(srcPath string, info os.FileInfo, err error) error {
		if err != nil || srcPath == srcRoot {
			return err
		}

		suffix := strings.TrimPrefix(srcPath, srcRoot)
		for len(suffix) > 0 && suffix[0] == filepath.Separator {
			suffix = suffix[1:]
		}
		dstPath := filepath.Join(dstRoot, suffix)

		perm := info.Mode() & os.ModePerm
		if info.Mode()&os.ModeSymlink != 0 {
			info, err = os.Stat(srcPath)
			if err != nil {
				return err
			}
			perm = info.Mode() & os.ModePerm
		}

		// Always copy directories (don't symlink them).
		// If we add a file in the overlay, we don't want to add it in the original.
		if info.IsDir() {
			return os.Mkdir(dstPath, perm)
		}

		// If the OS supports symlinks, use them instead of copying bytes.
		if err := os.Symlink(filepath.Join(symBase, suffix), dstPath); err == nil {
			return nil
		}

		// Otherwise, copy the bytes.
		src, err := os.Open(srcPath)
		if err != nil {
			return err
		}
		defer src.Close()

		dst, err := os.OpenFile(dstPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
		if err != nil {
			return err
		}

		_, err = io.Copy(dst, src)
		if closeErr := dst.Close(); err == nil {
			err = closeErr
		}
		return err
	})
}
