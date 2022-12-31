// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// The copyfiles script copies the contents of the internal cmd/go robustio
// package to the current directory, with adjustments to make it build.
//
// NOTE: In retrospect this script got out of hand, as we have to perform
// various operations on the package to get it to build at old Go versions. If
// in the future it proves to be flaky, delete it and just copy code manually.
package main

import (
	"bytes"
	"go/build/constraint"
	"go/scanner"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func main() {
	dir := filepath.Join(runtime.GOROOT(), "src", "cmd", "go", "internal", "robustio")

	entries, err := os.ReadDir(dir)
	if err != nil {
		log.Fatalf("reading the robustio dir: %v", err)
	}

	// Collect file content so that we can validate before copying.
	fileContent := make(map[string][]byte)
	windowsImport := []byte("\t\"internal/syscall/windows\"\n")
	foundWindowsImport := false
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".go") {
			pth := filepath.Join(dir, entry.Name())
			content, err := os.ReadFile(pth)
			if err != nil {
				log.Fatalf("reading %q: %v", entry.Name(), err)
			}

			// Replace the use of internal/syscall/windows.ERROR_SHARING_VIOLATION
			// with a local constant.
			if entry.Name() == "robustio_windows.go" && bytes.Contains(content, windowsImport) {
				foundWindowsImport = true
				content = bytes.Replace(content, windowsImport, nil, 1)
				content = bytes.Replace(content, []byte("windows.ERROR_SHARING_VIOLATION"), []byte("ERROR_SHARING_VIOLATION"), -1)
			}

			// Replace os.ReadFile with ioutil.ReadFile (for 1.15 and older). We
			// attempt to match calls (via the '('), to avoid matching mentions of
			// os.ReadFile in comments.
			//
			// TODO(rfindley): once we (shortly!) no longer support 1.15, remove
			// this and break the build.
			if bytes.Contains(content, []byte("os.ReadFile(")) {
				content = bytes.Replace(content, []byte("\"os\""), []byte("\"io/ioutil\"\n\t\"os\""), 1)
				content = bytes.Replace(content, []byte("os.ReadFile("), []byte("ioutil.ReadFile("), -1)
			}

			// Add +build constraints, for 1.16.
			content = addPlusBuildConstraints(content)

			fileContent[entry.Name()] = content
		}
	}

	if !foundWindowsImport {
		log.Fatal("missing expected import of internal/syscall/windows in robustio_windows.go")
	}

	for name, content := range fileContent {
		if err := os.WriteFile(name, content, 0644); err != nil {
			log.Fatalf("writing %q: %v", name, err)
		}
	}
}

// addPlusBuildConstraints splices in +build constraints for go:build
// constraints encountered in the source.
//
// Gopls still builds at Go 1.16, which requires +build constraints.
func addPlusBuildConstraints(src []byte) []byte {
	var s scanner.Scanner
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(src))
	s.Init(file, src, nil /* no error handler */, scanner.ScanComments)

	result := make([]byte, 0, len(src))
	lastInsertion := 0
	for {
		pos, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		if tok == token.COMMENT {
			if c, err := constraint.Parse(lit); err == nil {
				plusBuild, err := constraint.PlusBuildLines(c)
				if err != nil {
					log.Fatalf("computing +build constraint for %q: %v", lit, err)
				}
				insertAt := file.Offset(pos) + len(lit)
				result = append(result, src[lastInsertion:insertAt]...)
				result = append(result, []byte("\n"+strings.Join(plusBuild, "\n"))...)
				lastInsertion = insertAt
			}
		}
	}
	result = append(result, src[lastInsertion:]...)
	return result
}
