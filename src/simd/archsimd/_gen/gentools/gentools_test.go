// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gentools

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGoFileFormatting(t *testing.T) {
	t.Parallel()

	var outBuf bytes.Buffer
	var files Files
	files.Options = &Options{
		Txtar:  true,
		Output: &outBuf,
	}

	buf := files.NewGoFile("test.go")
	buf.WriteString("package test\nfunc Foo() int { return 42 }\n")

	err := files.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	outStr := outBuf.String()
	expected := "-- src/test.go --\npackage test\n\nfunc Foo() int { return 42 }\n"
	if outStr != expected {
		t.Errorf("got txtar output:\n%q\nwant:\n%q", outStr, expected)
	}
}

func TestFormattingError(t *testing.T) {
	t.Parallel()

	var errBuf bytes.Buffer
	var files Files
	files.Options = &Options{
		ErrOutput: &errBuf,
	}

	buf := files.NewGoFile("bad.go")
	buf.WriteString("package test\n\tfunc Foo( {\n") // invalid syntax with tab indentation

	err := files.Flush()
	if err == nil {
		t.Fatalf("expected formatting error, got nil")
	}
	if !strings.Contains(err.Error(), "error formatting bad.go") {
		t.Errorf("unexpected error message: %v", err)
	}

	const expectedErrOutput = "package test\n\tfunc Foo( {\n\t          ^\n2:12: expected ')', found '{'\n"
	if errBuf.String() != expectedErrOutput {
		t.Errorf("got error output:\n%q\nwant:\n%q", errBuf.String(), expectedErrOutput)
	}
}

func TestWriteMode(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	var files Files
	files.Options = &Options{
		GOROOT: tmpDir,
		Write:  true,
	}

	gobuf := files.NewGoFile("pkg/a.go")
	gobuf.WriteString("package pkg\nconst X = 1\n")

	const expectedRawContent = "raw content\n"
	rawbuf := files.NewRawFile("pkg/a.txt")
	rawbuf.WriteString(expectedRawContent)

	if err := files.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	aGo, err := os.ReadFile(filepath.Join(tmpDir, "src", "pkg", "a.go"))
	if err != nil {
		t.Fatalf("reading a.go: %v", err)
	}
	if string(aGo) != "package pkg\n\nconst X = 1\n" {
		t.Errorf("unexpected a.go content: %q", string(aGo))
	}

	aTxt, err := os.ReadFile(filepath.Join(tmpDir, "src", "pkg", "a.txt"))
	if err != nil {
		t.Fatalf("reading a.txt: %v", err)
	}
	if string(aTxt) != expectedRawContent {
		t.Errorf("unexpected a.txt content: %q", string(aTxt))
	}
}

func TestWriteAsideMode(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	tmpDir2 := t.TempDir()
	var files Files
	files.Options = &Options{
		GOROOT: tmpDir,
		outDir: tmpDir2,
		Write:  true,
	}

	gobuf := files.NewGoFile("pkg/a.go")
	gobuf.WriteString("package pkg\nconst X = 1\n")

	const expectedRawContent = "raw content\n"
	rawbuf := files.NewRawFile("pkg/a.txt")
	rawbuf.WriteString(expectedRawContent)

	if err := files.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	aGo, err := os.ReadFile(filepath.Join(tmpDir2, "src", "pkg", "a.go"))
	if err != nil {
		t.Fatalf("reading a.go: %v", err)
	}
	if string(aGo) != "package pkg\n\nconst X = 1\n" {
		t.Errorf("unexpected a.go content: %q", string(aGo))
	}

	aTxt, err := os.ReadFile(filepath.Join(tmpDir2, "src", "pkg", "a.txt"))
	if err != nil {
		t.Fatalf("reading a.txt: %v", err)
	}
	if string(aTxt) != expectedRawContent {
		t.Errorf("unexpected a.txt content: %q", string(aTxt))
	}
}

func TestWriteAsideRefusesOverwrite(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	outDir := t.TempDir()

	// Pre-create a file in the output directory
	existingFile := filepath.Join(outDir, "src", "pkg", "a.go")
	if err := os.MkdirAll(filepath.Dir(existingFile), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(existingFile, []byte("existing"), 0644); err != nil {
		t.Fatal(err)
	}

	var files Files
	files.Options = &Options{
		GOROOT: tmpDir,
		outDir: outDir,
		Write:  true,
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic when creating file that exists in output directory, got none")
		}
		msg, ok := r.(string)
		if !ok || !strings.Contains(msg, "file already exists in output directory") {
			t.Errorf("unexpected panic message: %v", r)
		}
	}()

	files.NewGoFile("pkg/a.go")
}

func TestDuplicateFileCreationPanics(t *testing.T) {
	t.Parallel()

	var files Files
	files.NewGoFile("pkg/a.go")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on duplicate file creation, got none")
		}
		msg, ok := r.(string)
		if !ok || !strings.Contains(msg, "generator already created file") {
			t.Errorf("unexpected panic message: %v", r)
		}
	}()

	files.NewGoFile("pkg/a.go")
}

func TestDiffMode(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	targetFile := filepath.Join(tmpDir, "src", "pkg", "a.go")
	os.MkdirAll(filepath.Dir(targetFile), 0755)
	os.WriteFile(targetFile, []byte("package pkg\n\nconst X = 1\n"), 0644)

	// Test matching content
	var files1 Files
	files1.Options = &Options{
		GOROOT: tmpDir,
		Diff:   true,
	}
	buf1 := files1.NewGoFile("pkg/a.go")
	buf1.WriteString("package pkg\nconst X = 1\n")
	if err := files1.Flush(); err != nil {
		t.Errorf("expected no diff error, got: %v", err)
	}

	// Test non-matching content
	var outBuf bytes.Buffer
	var files2 Files
	files2.Options = &Options{
		GOROOT: tmpDir,
		Diff:   true,
		Output: &outBuf,
	}
	buf2 := files2.NewGoFile("pkg/a.go")
	buf2.WriteString("package pkg\nconst X = 2\n")

	err := files2.Flush()
	if err == nil {
		t.Errorf("expected diff error, got nil")
	}

	if !strings.Contains(outBuf.String(), "-const X = 1") || !strings.Contains(outBuf.String(), "+const X = 2") {
		t.Errorf("unexpected diff output:\n%s", outBuf.String())
	}
}

func TestPostProcessor(t *testing.T) {
	t.Parallel()

	var outBuf bytes.Buffer
	var files Files
	files.Options = &Options{
		Txtar:  true,
		Output: &outBuf,
	}

	files.AddPostProcessor(func(relPath string, isGo bool, content []byte) ([]byte, error) {
		if !isGo {
			return append([]byte("# header\n"), content...), nil
		}
		// In Go files, replace Bar with Baz
		return bytes.ReplaceAll(content, []byte("Bar"), []byte("Baz")), nil
	})

	goBuf := files.NewGoFile("test.go")
	goBuf.WriteString("package test\nfunc Bar() int { return 1 }\n")

	rawBuf := files.NewRawFile("test.txt")
	rawBuf.WriteString("hello\n")

	if err := files.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	outStr := outBuf.String()
	expected := "-- src/test.go --\npackage test\n\nfunc Baz() int { return 1 }\n\n-- src/test.txt --\n# header\nhello\n"
	if outStr != expected {
		t.Errorf("got txtar output:\n%q\nwant:\n%q", outStr, expected)
	}
}

func TestPostProcessorError(t *testing.T) {
	t.Parallel()

	var files Files
	files.Options = &Options{
		Txtar:  true,
		Output: &bytes.Buffer{},
	}

	files.AddPostProcessor(func(relPath string, isGo bool, content []byte) ([]byte, error) {
		if relPath == "bad.go" {
			return nil, fmt.Errorf("custom hook error on %s", relPath)
		}
		return content, nil
	})

	buf := files.NewGoFile("bad.go")
	buf.WriteString("package test\n")

	err := files.Flush()
	if err == nil {
		t.Fatalf("expected post-processor error, got nil")
	}
	if !strings.Contains(err.Error(), "custom hook error on bad.go") {
		t.Errorf("unexpected error: %v", err)
	}
}
