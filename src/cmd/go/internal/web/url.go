// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package web

import (
	"errors"
	"net/url"
	"path/filepath"
	"strings"
)

// TODO(golang.org/issue/32456): If accepted, move these functions into the
// net/url package.

var errNotAbsolute = errors.New("path is not absolute")

func urlToFilePath(u *url.URL) (string, error) {
	if u.Scheme != "file" {
		return "", errors.New("non-file URL")
	}

	checkAbs := func(path string) (string, error) {
		if !filepath.IsAbs(path) {
			return "", errNotAbsolute
		}
		return path, nil
	}

	if u.Path == "" {
		if u.Host != "" || u.Opaque == "" {
			return "", errors.New("file URL missing path")
		}
		return checkAbs(filepath.FromSlash(u.Opaque))
	}

	path, err := convertFileURLPath(u.Host, u.Path)
	if err != nil {
		return path, err
	}
	return checkAbs(path)
}

func urlFromFilePath(path string) (*url.URL, error) {
	if !filepath.IsAbs(path) {
		return nil, errNotAbsolute
	}

	// If path has a Windows volume name, convert the volume to a host and prefix
	// per https://blogs.msdn.microsoft.com/ie/2006/12/06/file-uris-in-windows/.
	if vol := filepath.VolumeName(path); vol != "" {
		if strings.HasPrefix(vol, `\\`) {
			path = filepath.ToSlash(path[2:])
			i := strings.IndexByte(path, '/')

			if i < 0 {
				// A degenerate case.
				// \\host.example.com (without a share name)
				// becomes
				// file://host.example.com/
				return &url.URL{
					Scheme: "file",
					Host:   path,
					Path:   "/",
				}, nil
			}

			// \\host.example.com\Share\path\to\file
			// becomes
			// file://host.example.com/Share/path/to/file
			return &url.URL{
				Scheme: "file",
				Host:   path[:i],
				Path:   filepath.ToSlash(path[i:]),
			}, nil
		}

		// C:\path\to\file
		// becomes
		// file:///C:/path/to/file
		return &url.URL{
			Scheme: "file",
			Path:   "/" + filepath.ToSlash(path),
		}, nil
	}

	// /path/to/file
	// becomes
	// file:///path/to/file
	return &url.URL{
		Scheme: "file",
		Path:   filepath.ToSlash(path),
	}, nil
}
