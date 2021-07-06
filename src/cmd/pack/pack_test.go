// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"cmd/internal/archive"
	"fmt"
	"internal/testenv"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// testCreate creates an archive in the specified directory.
func testCreate(t *testing.T, dir string) {
	name := filepath.Join(dir, "pack.a")
	ar := openArchive(name, os.O_RDWR|os.O_CREATE, nil)
	// Add an entry by hand.
	ar.addFile(helloFile.Reset())
	ar.a.File().Close()
	// Now check it.
	ar = openArchive(name, os.O_RDONLY, []string{helloFile.name})
	var buf bytes.Buffer
	stdout = &buf
	verbose = true
	defer func() {
		stdout = os.Stdout
		verbose = false
	}()
	ar.scan(ar.printContents)
	ar.a.File().Close()
	result := buf.String()
	// Expect verbose output plus file contents.
	expect := fmt.Sprintf("%s\n%s", helloFile.name, helloFile.contents)
	if result != expect {
		t.Fatalf("expected %q got %q", expect, result)
	}
}

// Test that we can create an archive, write to it, and get the same contents back.
// Tests the rv and then the pv command on a new archive.
func TestCreate(t *testing.T) {
	dir := t.TempDir()
	testCreate(t, dir)
}

// Test that we can create an archive twice with the same name (Issue 8369).
func TestCreateTwice(t *testing.T) {
	dir := t.TempDir()
	testCreate(t, dir)
	testCreate(t, dir)
}

// Test that we can create an archive, put some files in it, and get back a correct listing.
// Tests the tv command.
func TestTableOfContents(t *testing.T) {
	dir := t.TempDir()
	name := filepath.Join(dir, "pack.a")
	ar := openArchive(name, os.O_RDWR|os.O_CREATE, nil)

	// Add some entries by hand.
	ar.addFile(helloFile.Reset())
	ar.addFile(goodbyeFile.Reset())
	ar.a.File().Close()

	// Now print it.
	var buf bytes.Buffer
	stdout = &buf
	verbose = true
	defer func() {
		stdout = os.Stdout
		verbose = false
	}()
	ar = openArchive(name, os.O_RDONLY, nil)
	ar.scan(ar.tableOfContents)
	ar.a.File().Close()
	result := buf.String()
	// Expect verbose listing.
	expect := fmt.Sprintf("%s\n%s\n", helloFile.Entry(), goodbyeFile.Entry())
	if result != expect {
		t.Fatalf("expected %q got %q", expect, result)
	}

	// Do it again without verbose.
	verbose = false
	buf.Reset()
	ar = openArchive(name, os.O_RDONLY, nil)
	ar.scan(ar.tableOfContents)
	ar.a.File().Close()
	result = buf.String()
	// Expect non-verbose listing.
	expect = fmt.Sprintf("%s\n%s\n", helloFile.name, goodbyeFile.name)
	if result != expect {
		t.Fatalf("expected %q got %q", expect, result)
	}

	// Do it again with file list arguments.
	verbose = false
	buf.Reset()
	ar = openArchive(name, os.O_RDONLY, []string{helloFile.name})
	ar.scan(ar.tableOfContents)
	ar.a.File().Close()
	result = buf.String()
	// Expect only helloFile.
	expect = fmt.Sprintf("%s\n", helloFile.name)
	if result != expect {
		t.Fatalf("expected %q got %q", expect, result)
	}
}

// Test that we can create an archive, put some files in it, and get back a file.
// Tests the x command.
func TestExtract(t *testing.T) {
	dir := t.TempDir()
	name := filepath.Join(dir, "pack.a")
	ar := openArchive(name, os.O_RDWR|os.O_CREATE, nil)
	// Add some entries by hand.
	ar.addFile(helloFile.Reset())
	ar.addFile(goodbyeFile.Reset())
	ar.a.File().Close()
	// Now extract one file. We chdir to the directory of the archive for simplicity.
	pwd, err := os.Getwd()
	if err != nil {
		t.Fatal("os.Getwd: ", err)
	}
	err = os.Chdir(dir)
	if err != nil {
		t.Fatal("os.Chdir: ", err)
	}
	defer func() {
		err := os.Chdir(pwd)
		if err != nil {
			t.Fatal("os.Chdir: ", err)
		}
	}()
	ar = openArchive(name, os.O_RDONLY, []string{goodbyeFile.name})
	ar.scan(ar.extractContents)
	ar.a.File().Close()
	data, err := os.ReadFile(goodbyeFile.name)
	if err != nil {
		t.Fatal(err)
	}
	// Expect contents of file.
	result := string(data)
	expect := goodbyeFile.contents
	if result != expect {
		t.Fatalf("expected %q got %q", expect, result)
	}
}

// Test that pack-created archives can be understood by the tools.
func TestHello(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()
	hello := filepath.Join(dir, "hello.go")
	prog := `
		package main
		func main() {
			println("hello world")
		}
	`
	err := os.WriteFile(hello, []byte(prog), 0666)
	if err != nil {
		t.Fatal(err)
	}

	run := func(args ...string) string {
		return doRun(t, dir, args...)
	}

	goBin := testenv.GoToolPath(t)
	run(goBin, "build", "cmd/pack") // writes pack binary to dir
	run(goBin, "tool", "compile", "hello.go")
	run("./pack", "grc", "hello.a", "hello.o")
	run(goBin, "tool", "link", "-o", "a.out", "hello.a")
	out := run("./a.out")
	if out != "hello world\n" {
		t.Fatalf("incorrect output: %q, want %q", out, "hello world\n")
	}
}

// Test that pack works with very long lines in PKGDEF.
func TestLargeDefs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()
	large := filepath.Join(dir, "large.go")
	f, err := os.Create(large)
	if err != nil {
		t.Fatal(err)
	}
	b := bufio.NewWriter(f)

	printf := func(format string, args ...interface{}) {
		_, err := fmt.Fprintf(b, format, args...)
		if err != nil {
			t.Fatalf("Writing to %s: %v", large, err)
		}
	}

	printf("package large\n\ntype T struct {\n")
	for i := 0; i < 1000; i++ {
		printf("f%d int `tag:\"", i)
		for j := 0; j < 100; j++ {
			printf("t%d=%d,", j, j)
		}
		printf("\"`\n")
	}
	printf("}\n")
	if err = b.Flush(); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}

	main := filepath.Join(dir, "main.go")
	prog := `
		package main
		import "large"
		var V large.T
		func main() {
			println("ok")
		}
	`
	err = os.WriteFile(main, []byte(prog), 0666)
	if err != nil {
		t.Fatal(err)
	}

	run := func(args ...string) string {
		return doRun(t, dir, args...)
	}

	goBin := testenv.GoToolPath(t)
	run(goBin, "build", "cmd/pack") // writes pack binary to dir
	run(goBin, "tool", "compile", "large.go")
	run("./pack", "grc", "large.a", "large.o")
	run(goBin, "tool", "compile", "-I", ".", "main.go")
	run(goBin, "tool", "link", "-L", ".", "-o", "a.out", "main.o")
	out := run("./a.out")
	if out != "ok\n" {
		t.Fatalf("incorrect output: %q, want %q", out, "ok\n")
	}
}

// Test that "\n!\n" inside export data doesn't result in a truncated
// package definition when creating a .a archive from a .o Go object.
func TestIssue21703(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()

	const aSrc = `package a; const X = "\n!\n"`
	err := os.WriteFile(filepath.Join(dir, "a.go"), []byte(aSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	const bSrc = `package b; import _ "a"`
	err = os.WriteFile(filepath.Join(dir, "b.go"), []byte(bSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	run := func(args ...string) string {
		return doRun(t, dir, args...)
	}

	goBin := testenv.GoToolPath(t)
	run(goBin, "build", "cmd/pack") // writes pack binary to dir
	run(goBin, "tool", "compile", "a.go")
	run("./pack", "c", "a.a", "a.o")
	run(goBin, "tool", "compile", "-I", ".", "b.go")
}

// Test the "c" command can "see through" the archive generated by the compiler.
// This is peculiar. (See issue #43271)
func TestCreateWithCompilerObj(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()
	src := filepath.Join(dir, "p.go")
	prog := "package p; var X = 42\n"
	err := os.WriteFile(src, []byte(prog), 0666)
	if err != nil {
		t.Fatal(err)
	}

	run := func(args ...string) string {
		return doRun(t, dir, args...)
	}

	goBin := testenv.GoToolPath(t)
	run(goBin, "build", "cmd/pack") // writes pack binary to dir
	run(goBin, "tool", "compile", "-pack", "-o", "p.a", "p.go")
	run("./pack", "c", "packed.a", "p.a")
	fi, err := os.Stat(filepath.Join(dir, "p.a"))
	if err != nil {
		t.Fatalf("stat p.a failed: %v", err)
	}
	fi2, err := os.Stat(filepath.Join(dir, "packed.a"))
	if err != nil {
		t.Fatalf("stat packed.a failed: %v", err)
	}
	// For compiler-generated object file, the "c" command is
	// expected to get (essentially) the same file back, instead
	// of packing it into a new archive with a single entry.
	if want, got := fi.Size(), fi2.Size(); want != got {
		t.Errorf("packed file with different size: want %d, got %d", want, got)
	}

	// Test -linkobj flag as well.
	run(goBin, "tool", "compile", "-linkobj", "p2.a", "-o", "p.x", "p.go")
	run("./pack", "c", "packed2.a", "p2.a")
	fi, err = os.Stat(filepath.Join(dir, "p2.a"))
	if err != nil {
		t.Fatalf("stat p2.a failed: %v", err)
	}
	fi2, err = os.Stat(filepath.Join(dir, "packed2.a"))
	if err != nil {
		t.Fatalf("stat packed2.a failed: %v", err)
	}
	if want, got := fi.Size(), fi2.Size(); want != got {
		t.Errorf("packed file with different size: want %d, got %d", want, got)
	}

	run("./pack", "c", "packed3.a", "p.x")
	fi, err = os.Stat(filepath.Join(dir, "p.x"))
	if err != nil {
		t.Fatalf("stat p.x failed: %v", err)
	}
	fi2, err = os.Stat(filepath.Join(dir, "packed3.a"))
	if err != nil {
		t.Fatalf("stat packed3.a failed: %v", err)
	}
	if want, got := fi.Size(), fi2.Size(); want != got {
		t.Errorf("packed file with different size: want %d, got %d", want, got)
	}
}

// Test the "r" command creates the output file if it does not exist.
func TestRWithNonexistentFile(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	dir := t.TempDir()
	src := filepath.Join(dir, "p.go")
	prog := "package p; var X = 42\n"
	err := os.WriteFile(src, []byte(prog), 0666)
	if err != nil {
		t.Fatal(err)
	}

	run := func(args ...string) string {
		return doRun(t, dir, args...)
	}

	goBin := testenv.GoToolPath(t)
	run(goBin, "build", "cmd/pack") // writes pack binary to dir
	run(goBin, "tool", "compile", "-o", "p.o", "p.go")
	run("./pack", "r", "p.a", "p.o") // should succeed
}

// doRun runs a program in a directory and returns the output.
func doRun(t *testing.T, dir string, args ...string) string {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %v\n%s", args, err, string(out))
	}
	return string(out)
}

// Fake implementation of files.

var helloFile = &FakeFile{
	name:     "hello",
	contents: "hello world", // 11 bytes, an odd number.
	mode:     0644,
}

var goodbyeFile = &FakeFile{
	name:     "goodbye",
	contents: "Sayonara, Jim", // 13 bytes, another odd number.
	mode:     0644,
}

// FakeFile implements FileLike and also fs.FileInfo.
type FakeFile struct {
	name     string
	contents string
	mode     fs.FileMode
	offset   int
}

// Reset prepares a FakeFile for reuse.
func (f *FakeFile) Reset() *FakeFile {
	f.offset = 0
	return f
}

// FileLike methods.

func (f *FakeFile) Name() string {
	// A bit of a cheat: we only have a basename, so that's also ok for FileInfo.
	return f.name
}

func (f *FakeFile) Stat() (fs.FileInfo, error) {
	return f, nil
}

func (f *FakeFile) Read(p []byte) (int, error) {
	if f.offset >= len(f.contents) {
		return 0, io.EOF
	}
	n := copy(p, f.contents[f.offset:])
	f.offset += n
	return n, nil
}

func (f *FakeFile) Close() error {
	return nil
}

// fs.FileInfo methods.

func (f *FakeFile) Size() int64 {
	return int64(len(f.contents))
}

func (f *FakeFile) Mode() fs.FileMode {
	return f.mode
}

func (f *FakeFile) ModTime() time.Time {
	return time.Time{}
}

func (f *FakeFile) IsDir() bool {
	return false
}

func (f *FakeFile) Sys() interface{} {
	return nil
}

// Special helpers.

func (f *FakeFile) Entry() *archive.Entry {
	return &archive.Entry{
		Name:  f.name,
		Mtime: 0, // Defined to be zero.
		Uid:   0, // Ditto.
		Gid:   0, // Ditto.
		Mode:  f.mode,
		Data:  archive.Data{Size: int64(len(f.contents))},
	}
}
