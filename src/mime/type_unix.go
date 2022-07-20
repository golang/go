// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm)

package mime

import (
	"bufio"
	"bytes"
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
		// Each line should be of format: weight:mimetype:glob[:morefields...]
		fields := strings.Split(scanner.Text(), ":")
		if len(fields) < 3 || len(fields[0]) < 1 || len(fields[2]) < 3 {
			continue
		} else if fields[0][0] == '#' || fields[2][0] != '*' || fields[2][1] != '.' {
			continue
		}

		extension := fields[2][1:]
		switch {
		case strings.ContainsAny(extension, "?*"):
			// Not a bare extension, but a glob. Ignore for now:
			// - we do not have an implementation for this glob
			//   syntax (translation to path/filepath.Match could
			//   be possible)
			// - support for globs with weight ordering would have
			//   performance impact to all lookups to support the
			//   rarely seen glob entries
			// - trying to match glob metacharacters literally is
			//   not useful
			continue
		case strings.Contains(extension, "["):
			if extensions, ok := expand(extension); ok {
				for i := range extensions {
					setExtensionType(extensions[i], fields[1])
				}
			}
			continue
		}

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

func expand(glob string) ([]string, bool) {
	runes := []rune(glob)
	resultSize := 1
	stringSize := 0

countLoop:
	for i := 0; i < len(runes); i++ {
		switch runes[i] {
		case '[':
			for j := i + 1; j < len(runes); j++ {
				if runes[j] == ']' {
					i = j
					continue countLoop
				}
				if runes[j+1] == '-' {
					if j+2 >= len(runes) {
						return nil, false
					}
					resultSize *= int(runes[j+2]-runes[j]) + 1
					stringSize++
					j += 2
					continue
				}
				resultSize++
				stringSize++
			}
		default:
			stringSize++
		}
	}

	buffers := make([]bytes.Buffer, resultSize, resultSize)
	for i := range buffers {
		buffers[i].Grow(stringSize)
	}

	for i := 0; i < len(runes); i++ {
		switch runes[i] {
		case '[':
			var expanded []rune
			for j := i + 1; j < len(runes); j++ {
				if runes[j] == ']' {
					i = j
					break
				}
				if runes[j+1] == '-' {
					for k := runes[j]; k <= runes[j+2]; k++ {
						expanded = append(expanded, k)
					}
					j += 2
					continue
				}
				expanded = append(expanded, runes[j])
			}

			for j, k := 0, 0; j < resultSize; j, k = j+1, (k+1)%len(expanded) {
				buffers[j].WriteRune(expanded[k])
			}

		default:
			for j := 0; j < resultSize; j++ {
				buffers[j].WriteRune(runes[i])
			}
		}
	}

	result := make([]string, 0, resultSize)
	for i := 0; i < resultSize; i++ {
		result = append(result, buffers[i].String())
	}

	return result, true

}
