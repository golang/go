// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || (js && wasm) || linux || netbsd || openbsd || solaris

package mime

import (
	"bufio"
	"os"
	"strings"
)

func init() {
	osInitMime = initMimeUnix
}

// See https://specifications.freedesktop.org/shared-mime-info-spec/shared-mime-info-spec-0.21.html
// for the FreeDesktop Shared MIME-info Database specification.
var mimeGlobs = []string{
	"/usr/local/share/mime/globs2",
	"/usr/share/mime/globs2",
}

// Common locations for mime.types files on unix.
var typeFiles = []string{
	"/etc/mime.types",
	"/etc/apache2/mime.types",
	"/etc/apache/mime.types",
	"/etc/httpd/conf/mime.types",
}

func loadMimeGlobsFile(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		// Each line should be of format: weight:mimetype:*.ext
		fields := strings.Split(scanner.Text(), ":")
		if len(fields) < 3 || len(fields[0]) < 1 || len(fields[2]) < 2 {
			continue
		} else if fields[0][0] == '#' || fields[2][0] != '*' {
			continue
		}

		extension := fields[2][1:]
		if _, ok := mimeTypes.Load(extension); ok {
			// We've already seen this extension.
			// The file is in weight order, so we keep
			// the first entry that we see.
			continue
		}

		setExtensionType(extension, fields[1])
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	return nil
}

func loadMimeFile(filename string) {
	f, err := os.Open(filename)
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) <= 1 || fields[0][0] == '#' {
			continue
		}
		mimeType := fields[0]
		for _, ext := range fields[1:] {
			if ext[0] == '#' {
				break
			}
			setExtensionType("."+ext, mimeType)
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
}

func initMimeUnix() {
	for _, filename := range mimeGlobs {
		if err := loadMimeGlobsFile(filename); err == nil {
			return // Stop checking more files if mimetype database is found.
		}
	}

	// Fallback if no system-generated mimetype database exists.
	for _, filename := range typeFiles {
		loadMimeFile(filename)
	}
}

func initMimeForTests() map[string]string {
	mimeGlobs = []string{""}
	typeFiles = []string{"testdata/test.types"}
	return map[string]string{
		".T1":  "application/test",
		".t2":  "text/test; charset=utf-8",
		".png": "image/png",
	}
}
