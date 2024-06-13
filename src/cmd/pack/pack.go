// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/archive"
	"cmd/internal/telemetry"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
)

const usageMessage = `Usage: pack op file.a [name....]
Where op is one of cprtx optionally followed by v for verbose output.
For compatibility with old Go build environments the op string grc is
accepted as a synonym for c.

For more information, run
	go doc cmd/pack`

func usage() {
	fmt.Fprintln(os.Stderr, usageMessage)
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("pack: ")
	telemetry.OpenCounters()
	// need "pack op archive" at least.
	if len(os.Args) < 3 {
		log.Print("not enough arguments")
		fmt.Fprintln(os.Stderr)
		usage()
	}
	setOp(os.Args[1])
	telemetry.Inc("pack/invocations")
	telemetry.Inc("pack/op:" + string(op))
	var ar *Archive
	switch op {
	case 'p':
		ar = openArchive(os.Args[2], os.O_RDONLY, os.Args[3:])
		ar.scan(ar.printContents)
	case 'r':
		ar = openArchive(os.Args[2], os.O_RDWR|os.O_CREATE, os.Args[3:])
		ar.addFiles()
	case 'c':
		ar = openArchive(os.Args[2], os.O_RDWR|os.O_TRUNC|os.O_CREATE, os.Args[3:])
		ar.addPkgdef()
		ar.addFiles()
	case 't':
		ar = openArchive(os.Args[2], os.O_RDONLY, os.Args[3:])
		ar.scan(ar.tableOfContents)
	case 'x':
		ar = openArchive(os.Args[2], os.O_RDONLY, os.Args[3:])
		ar.scan(ar.extractContents)
	default:
		log.Printf("invalid operation %q", os.Args[1])
		fmt.Fprintln(os.Stderr)
		usage()
	}
	if len(ar.files) > 0 {
		log.Fatalf("file %q not in archive", ar.files[0])
	}
}

// The unusual ancestry means the arguments are not Go-standard.
// These variables hold the decoded operation specified by the first argument.
// op holds the operation we are doing (prtx).
// verbose tells whether the 'v' option was specified.
var (
	op      rune
	verbose bool
)

// setOp parses the operation string (first argument).
func setOp(arg string) {
	// Recognize 'go tool pack grc' because that was the
	// formerly canonical way to build a new archive
	// from a set of input files. Accepting it keeps old
	// build systems working with both Go 1.2 and Go 1.3.
	if arg == "grc" {
		arg = "c"
	}

	for _, r := range arg {
		switch r {
		case 'c', 'p', 'r', 't', 'x':
			if op != 0 {
				// At most one can be set.
				usage()
			}
			op = r
		case 'v':
			if verbose {
				// Can be set only once.
				usage()
			}
			verbose = true
		default:
			usage()
		}
	}
}

const (
	arHeader = "!<arch>\n"
)

// An Archive represents an open archive file. It is always scanned sequentially
// from start to end, without backing up.
type Archive struct {
	a        *archive.Archive
	files    []string // Explicit list of files to be processed.
	pad      int      // Padding bytes required at end of current archive file
	matchAll bool     // match all files in archive
}

// archive opens (and if necessary creates) the named archive.
func openArchive(name string, mode int, files []string) *Archive {
	f, err := os.OpenFile(name, mode, 0666)
	if err != nil {
		log.Fatal(err)
	}
	var a *archive.Archive
	if mode&os.O_TRUNC != 0 { // the c command
		a, err = archive.New(f)
	} else {
		a, err = archive.Parse(f, verbose)
		if err != nil && mode&os.O_CREATE != 0 { // the r command
			a, err = archive.New(f)
		}
	}
	if err != nil {
		log.Fatal(err)
	}
	return &Archive{
		a:        a,
		files:    files,
		matchAll: len(files) == 0,
	}
}

// scan scans the archive and executes the specified action on each entry.
func (ar *Archive) scan(action func(*archive.Entry)) {
	for i := range ar.a.Entries {
		e := &ar.a.Entries[i]
		action(e)
	}
}

// listEntry prints to standard output a line describing the entry.
func listEntry(e *archive.Entry, verbose bool) {
	if verbose {
		fmt.Fprintf(stdout, "%s\n", e.String())
	} else {
		fmt.Fprintf(stdout, "%s\n", e.Name)
	}
}

// output copies the entry to the specified writer.
func (ar *Archive) output(e *archive.Entry, w io.Writer) {
	r := io.NewSectionReader(ar.a.File(), e.Offset, e.Size)
	n, err := io.Copy(w, r)
	if err != nil {
		log.Fatal(err)
	}
	if n != e.Size {
		log.Fatal("short file")
	}
}

// match reports whether the entry matches the argument list.
// If it does, it also drops the file from the to-be-processed list.
func (ar *Archive) match(e *archive.Entry) bool {
	if ar.matchAll {
		return true
	}
	for i, name := range ar.files {
		if e.Name == name {
			copy(ar.files[i:], ar.files[i+1:])
			ar.files = ar.files[:len(ar.files)-1]
			return true
		}
	}
	return false
}

// addFiles adds files to the archive. The archive is known to be
// sane and we are positioned at the end. No attempt is made
// to check for existing files.
func (ar *Archive) addFiles() {
	if len(ar.files) == 0 {
		usage()
	}
	for _, file := range ar.files {
		if verbose {
			fmt.Printf("%s\n", file)
		}

		f, err := os.Open(file)
		if err != nil {
			log.Fatal(err)
		}
		aro, err := archive.Parse(f, false)
		if err != nil || !isGoCompilerObjFile(aro) {
			f.Seek(0, io.SeekStart)
			ar.addFile(f)
			goto close
		}

		for _, e := range aro.Entries {
			if e.Type != archive.EntryGoObj || e.Name != "_go_.o" {
				continue
			}
			ar.a.AddEntry(archive.EntryGoObj, filepath.Base(file), 0, 0, 0, 0644, e.Size, io.NewSectionReader(f, e.Offset, e.Size))
		}
	close:
		f.Close()
	}
	ar.files = nil
}

// FileLike abstracts the few methods we need, so we can test without needing real files.
type FileLike interface {
	Name() string
	Stat() (fs.FileInfo, error)
	Read([]byte) (int, error)
	Close() error
}

// addFile adds a single file to the archive
func (ar *Archive) addFile(fd FileLike) {
	// Format the entry.
	// First, get its info.
	info, err := fd.Stat()
	if err != nil {
		log.Fatal(err)
	}
	// mtime, uid, gid are all zero so repeated builds produce identical output.
	mtime := int64(0)
	uid := 0
	gid := 0
	ar.a.AddEntry(archive.EntryNativeObj, info.Name(), mtime, uid, gid, info.Mode(), info.Size(), fd)
}

// addPkgdef adds the __.PKGDEF file to the archive, copied
// from the first Go object file on the file list, if any.
// The archive is known to be empty.
func (ar *Archive) addPkgdef() {
	done := false
	for _, file := range ar.files {
		f, err := os.Open(file)
		if err != nil {
			log.Fatal(err)
		}
		aro, err := archive.Parse(f, false)
		if err != nil || !isGoCompilerObjFile(aro) {
			goto close
		}

		for _, e := range aro.Entries {
			if e.Type != archive.EntryPkgDef {
				continue
			}
			if verbose {
				fmt.Printf("__.PKGDEF # %s\n", file)
			}
			ar.a.AddEntry(archive.EntryPkgDef, "__.PKGDEF", 0, 0, 0, 0644, e.Size, io.NewSectionReader(f, e.Offset, e.Size))
			done = true
		}
	close:
		f.Close()
		if done {
			break
		}
	}
}

// Finally, the actual commands. Each is an action.

// can be modified for testing.
var stdout io.Writer = os.Stdout

// printContents implements the 'p' command.
func (ar *Archive) printContents(e *archive.Entry) {
	ar.extractContents1(e, stdout)
}

// tableOfContents implements the 't' command.
func (ar *Archive) tableOfContents(e *archive.Entry) {
	if ar.match(e) {
		listEntry(e, verbose)
	}
}

// extractContents implements the 'x' command.
func (ar *Archive) extractContents(e *archive.Entry) {
	ar.extractContents1(e, nil)
}

func (ar *Archive) extractContents1(e *archive.Entry, out io.Writer) {
	if ar.match(e) {
		if verbose {
			listEntry(e, false)
		}
		if out == nil {
			f, err := os.OpenFile(e.Name, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0444 /*e.Mode*/)
			if err != nil {
				log.Fatal(err)
			}
			defer f.Close()
			out = f
		}
		ar.output(e, out)
	}
}

// isGoCompilerObjFile reports whether file is an object file created
// by the Go compiler, which is an archive file with exactly one entry
// of __.PKGDEF, or _go_.o, or both entries.
func isGoCompilerObjFile(a *archive.Archive) bool {
	switch len(a.Entries) {
	case 1:
		return (a.Entries[0].Type == archive.EntryGoObj && a.Entries[0].Name == "_go_.o") ||
			(a.Entries[0].Type == archive.EntryPkgDef && a.Entries[0].Name == "__.PKGDEF")
	case 2:
		var foundPkgDef, foundGo bool
		for _, e := range a.Entries {
			if e.Type == archive.EntryPkgDef && e.Name == "__.PKGDEF" {
				foundPkgDef = true
			}
			if e.Type == archive.EntryGoObj && e.Name == "_go_.o" {
				foundGo = true
			}
		}
		return foundPkgDef && foundGo
	default:
		return false
	}
}
