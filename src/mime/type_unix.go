// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package mime

import (
	"bufio"
	"os"
	"strings"
	"unicode"
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
	openingBracketIndex := -1
	closingBracketIndex := -1

	var prefix []byte
	var suffix []byte
	var mux *[]byte = &prefix

	for i, c := range glob {
		if c > unicode.MaxASCII {
			return nil, false
		}
		switch c {
		case '[':
			if len(*mux) > 0 && (*mux)[len(*mux)-1] == '\\' {
				(*mux)[len(*mux)-1] = glob[i]
				continue
			}
			if openingBracketIndex != -1 {
				if closingBracketIndex != -1 {
					return nil, false
				}
				continue
			}
			openingBracketIndex = i
			mux = &suffix
		case ']':
			if openingBracketIndex == -1 {
				*mux = append(*mux, ']')
				continue
			}
			if i == openingBracketIndex+1 {
				continue
			}
			closingBracketIndex = i
		default:
			if openingBracketIndex > -1 && closingBracketIndex == -1 {
				continue
			}
			*mux = append(*mux, glob[i])
		}
	}

	switch {
	case openingBracketIndex == -1 && closingBracketIndex == -1:
		return []string{string(prefix)}, true

	case openingBracketIndex != -1 && closingBracketIndex == -1:
		return []string{string(prefix) + glob[openingBracketIndex:]}, true

	case openingBracketIndex != -1 && openingBracketIndex+1 == '!':
		return nil, false
	}

	expansion := expandRangeWithoutNegation(glob[openingBracketIndex+1 : closingBracketIndex])
	if expansion == nil {
		return nil, false
	}

	results := make([]string, len(expansion))
	for i := 0; i < len(expansion); i++ {
		results[i] = string(prefix) + string(expansion[i]) + string(suffix)
	}

	return results, true
}

func expandRangeWithoutNegation(r string) []byte {
	var expansion []byte
	for i := 0; i < len(r); i++ {
		if r[i] == '!' && i == 0 {
			// no negations of range expression
			return nil
		}

		if r[i] != '-' {
			expansion = append(expansion, r[i])
			continue
		}

		if i == 0 || i == len(r)-1 {
			expansion = append(expansion, '-')
			continue
		}
		if r[i+1] < r[i-1] {
			// invalid character range
			return nil
		}

		for c := r[i-1] + 1; c <= r[i+1]; c++ {
			if c == '/' {
				// '/' cannot be matched: https://man7.org/linux/man-pages/man7/glob.7.html
				continue
			}
			expansion = append(expansion, c)
		}
		i++
	}
	return expansion
}
