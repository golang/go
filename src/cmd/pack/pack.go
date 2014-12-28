// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

/*
The archive format is:

First, on a line by itself
	!<arch>

Then zero or more file records. Each file record has a fixed-size one-line header
followed by data bytes followed by an optional padding byte. The header is:

	%-16s%-12d%-6d%-6d%-8o%-10d`
	name mtime uid gid mode size

(note the trailing backquote). The %-16s here means at most 16 *bytes* of
the name, and if shorter, space padded on the right.
*/

const usageMessage = `Usage: pack op file.a [name....]
Where op is one of cprtx optionally followed by v for verbose output.
For compatibility with old Go build environments the op string grc is
accepted as a synonym for c.

For more information, run
	godoc cmd/pack`

func usage() {
	fmt.Fprintln(os.Stderr, usageMessage)
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("pack: ")
	// need "pack op archive" at least.
	if len(os.Args) < 3 {
		log.Print("not enough arguments")
		fmt.Fprintln(os.Stderr)
		usage()
	}
	setOp(os.Args[1])
	var ar *Archive
	switch op {
	case 'p':
		ar = archive(os.Args[2], os.O_RDONLY, os.Args[3:])
		ar.scan(ar.printContents)
	case 'r':
		ar = archive(os.Args[2], os.O_RDWR, os.Args[3:])
		ar.scan(ar.skipContents)
		ar.addFiles()
	case 'c':
		ar = archive(os.Args[2], os.O_RDWR|os.O_TRUNC, os.Args[3:])
		ar.addPkgdef()
		ar.addFiles()
	case 't':
		ar = archive(os.Args[2], os.O_RDONLY, os.Args[3:])
		ar.scan(ar.tableOfContents)
	case 'x':
		ar = archive(os.Args[2], os.O_RDONLY, os.Args[3:])
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
	arHeader    = "!<arch>\n"
	entryHeader = "%s%-12d%-6d%-6d%-8o%-10d`\n"
	// In entryHeader the first entry, the name, is always printed as 16 bytes right-padded.
	entryLen   = 16 + 12 + 6 + 6 + 8 + 10 + 1 + 1
	timeFormat = "Jan _2 15:04 2006"
)

// An Archive represents an open archive file. It is always scanned sequentially
// from start to end, without backing up.
type Archive struct {
	fd       *os.File // Open file descriptor.
	files    []string // Explicit list of files to be processed.
	pad      int      // Padding bytes required at end of current archive file
	matchAll bool     // match all files in archive
}

// archive opens (and if necessary creates) the named archive.
func archive(name string, mode int, files []string) *Archive {
	// If the file exists, it must be an archive. If it doesn't exist, or if
	// we're doing the c command, indicated by O_TRUNC, truncate the archive.
	if !existingArchive(name) || mode&os.O_TRUNC != 0 {
		create(name)
		mode &^= os.O_TRUNC
	}
	fd, err := os.OpenFile(name, mode, 0)
	if err != nil {
		log.Fatal(err)
	}
	checkHeader(fd)
	return &Archive{
		fd:       fd,
		files:    files,
		matchAll: len(files) == 0,
	}
}

// create creates and initializes an archive that does not exist.
func create(name string) {
	fd, err := os.Create(name)
	if err != nil {
		log.Fatal(err)
	}
	_, err = fmt.Fprint(fd, arHeader)
	if err != nil {
		log.Fatal(err)
	}
	fd.Close()
}

// existingArchive reports whether the file exists and is a valid archive.
// If it exists but is not an archive, existingArchive will exit.
func existingArchive(name string) bool {
	fd, err := os.Open(name)
	if err != nil {
		if os.IsNotExist(err) {
			return false
		}
		log.Fatalf("cannot open file: %s", err)
	}
	checkHeader(fd)
	fd.Close()
	return true
}

// checkHeader verifies the header of the file. It assumes the file
// is positioned at 0 and leaves it positioned at the end of the header.
func checkHeader(fd *os.File) {
	buf := make([]byte, len(arHeader))
	_, err := io.ReadFull(fd, buf)
	if err != nil || string(buf) != arHeader {
		log.Fatalf("%s is not an archive: bad header", fd.Name())
	}
}

// An Entry is the internal representation of the per-file header information of one entry in the archive.
type Entry struct {
	name  string
	mtime int64
	uid   int
	gid   int
	mode  os.FileMode
	size  int64
}

func (e *Entry) String() string {
	return fmt.Sprintf("%s %6d/%-6d %12d %s %s",
		(e.mode & 0777).String(),
		e.uid,
		e.gid,
		e.size,
		time.Unix(e.mtime, 0).Format(timeFormat),
		e.name)
}

// readMetadata reads and parses the metadata for the next entry in the archive.
func (ar *Archive) readMetadata() *Entry {
	buf := make([]byte, entryLen)
	_, err := io.ReadFull(ar.fd, buf)
	if err == io.EOF {
		// No entries left.
		return nil
	}
	if err != nil || buf[entryLen-2] != '`' || buf[entryLen-1] != '\n' {
		log.Fatal("file is not an archive: bad entry")
	}
	entry := new(Entry)
	entry.name = strings.TrimRight(string(buf[:16]), " ")
	if len(entry.name) == 0 {
		log.Fatal("file is not an archive: bad name")
	}
	buf = buf[16:]
	str := string(buf)
	get := func(width, base, bitsize int) int64 {
		v, err := strconv.ParseInt(strings.TrimRight(str[:width], " "), base, bitsize)
		if err != nil {
			log.Fatal("file is not an archive: bad number in entry: ", err)
		}
		str = str[width:]
		return v
	}
	// %-16s%-12d%-6d%-6d%-8o%-10d`
	entry.mtime = get(12, 10, 64)
	entry.uid = int(get(6, 10, 32))
	entry.gid = int(get(6, 10, 32))
	entry.mode = os.FileMode(get(8, 8, 32))
	entry.size = get(10, 10, 64)
	return entry
}

// scan scans the archive and executes the specified action on each entry.
// When action returns, the file offset is at the start of the next entry.
func (ar *Archive) scan(action func(*Entry)) {
	for {
		entry := ar.readMetadata()
		if entry == nil {
			break
		}
		action(entry)
	}
}

// listEntry prints to standard output a line describing the entry.
func listEntry(ar *Archive, entry *Entry, verbose bool) {
	if verbose {
		fmt.Fprintf(stdout, "%s\n", entry)
	} else {
		fmt.Fprintf(stdout, "%s\n", entry.name)
	}
}

// output copies the entry to the specified writer.
func (ar *Archive) output(entry *Entry, w io.Writer) {
	n, err := io.Copy(w, io.LimitReader(ar.fd, entry.size))
	if err != nil {
		log.Fatal(err)
	}
	if n != entry.size {
		log.Fatal("short file")
	}
	if entry.size&1 == 1 {
		_, err := ar.fd.Seek(1, 1)
		if err != nil {
			log.Fatal(err)
		}
	}
}

// skip skips the entry without reading it.
func (ar *Archive) skip(entry *Entry) {
	size := entry.size
	if size&1 == 1 {
		size++
	}
	_, err := ar.fd.Seek(size, 1)
	if err != nil {
		log.Fatal(err)
	}
}

// match reports whether the entry matches the argument list.
// If it does, it also drops the file from the to-be-processed list.
func (ar *Archive) match(entry *Entry) bool {
	if ar.matchAll {
		return true
	}
	for i, name := range ar.files {
		if entry.name == name {
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
		fd, err := os.Open(file)
		if err != nil {
			log.Fatal(err)
		}
		ar.addFile(fd)
	}
	ar.files = nil
}

// FileLike abstracts the few methods we need, so we can test without needing real files.
type FileLike interface {
	Name() string
	Stat() (os.FileInfo, error)
	Read([]byte) (int, error)
	Close() error
}

// addFile adds a single file to the archive
func (ar *Archive) addFile(fd FileLike) {
	defer fd.Close()
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
	ar.startFile(info.Name(), mtime, uid, gid, info.Mode(), info.Size())
	n64, err := io.Copy(ar.fd, fd)
	if err != nil {
		log.Fatal("writing file: ", err)
	}
	if n64 != info.Size() {
		log.Fatalf("writing file: wrote %d bytes; file is size %d", n64, info.Size())
	}
	ar.endFile()
}

// startFile writes the archive entry header.
func (ar *Archive) startFile(name string, mtime int64, uid, gid int, mode os.FileMode, size int64) {
	n, err := fmt.Fprintf(ar.fd, entryHeader, exactly16Bytes(name), mtime, uid, gid, mode, size)
	if err != nil || n != entryLen {
		log.Fatal("writing entry header: ", err)
	}
	ar.pad = int(size & 1)
}

// endFile writes the archive entry tail (a single byte of padding, if the file size was odd).
func (ar *Archive) endFile() {
	if ar.pad != 0 {
		_, err := ar.fd.Write([]byte{0})
		if err != nil {
			log.Fatal("writing archive: ", err)
		}
		ar.pad = 0
	}
}

// addPkgdef adds the __.PKGDEF file to the archive, copied
// from the first Go object file on the file list, if any.
// The archive is known to be empty.
func (ar *Archive) addPkgdef() {
	for _, file := range ar.files {
		pkgdef, err := readPkgdef(file)
		if err != nil {
			continue
		}
		if verbose {
			fmt.Printf("__.PKGDEF # %s\n", file)
		}
		ar.startFile("__.PKGDEF", 0, 0, 0, 0644, int64(len(pkgdef)))
		_, err = ar.fd.Write(pkgdef)
		if err != nil {
			log.Fatal("writing __.PKGDEF: ", err)
		}
		ar.endFile()
		break
	}
}

// readPkgdef extracts the __.PKGDEF data from a Go object file.
func readPkgdef(file string) (data []byte, err error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read from file, collecting header for __.PKGDEF.
	// The header is from the beginning of the file until a line
	// containing just "!". The first line must begin with "go object ".
	rbuf := bufio.NewReader(f)
	var wbuf bytes.Buffer
	for {
		line, err := rbuf.ReadBytes('\n')
		if err != nil {
			return nil, err
		}
		if wbuf.Len() == 0 && !bytes.HasPrefix(line, []byte("go object ")) {
			return nil, errors.New("not a Go object file")
		}
		if bytes.Equal(line, []byte("!\n")) {
			break
		}
		wbuf.Write(line)
	}
	return wbuf.Bytes(), nil
}

// exactly16Bytes truncates the string if necessary so it is at most 16 bytes long,
// then pads the result with spaces to be exactly 16 bytes.
// Fmt uses runes for its width calculation, but we need bytes in the entry header.
func exactly16Bytes(s string) string {
	for len(s) > 16 {
		_, wid := utf8.DecodeLastRuneInString(s)
		s = s[:len(s)-wid]
	}
	const sixteenSpaces = "                "
	s += sixteenSpaces[:16-len(s)]
	return s
}

// Finally, the actual commands. Each is an action.

// can be modified for testing.
var stdout io.Writer = os.Stdout

// printContents implements the 'p' command.
func (ar *Archive) printContents(entry *Entry) {
	if ar.match(entry) {
		if verbose {
			listEntry(ar, entry, false)
		}
		ar.output(entry, stdout)
	} else {
		ar.skip(entry)
	}
}

// skipContents implements the first part of the 'r' command.
// It just scans the archive to make sure it's intact.
func (ar *Archive) skipContents(entry *Entry) {
	ar.skip(entry)
}

// tableOfContents implements the 't' command.
func (ar *Archive) tableOfContents(entry *Entry) {
	if ar.match(entry) {
		listEntry(ar, entry, verbose)
	}
	ar.skip(entry)
}

// extractContents implements the 'x' command.
func (ar *Archive) extractContents(entry *Entry) {
	if ar.match(entry) {
		if verbose {
			listEntry(ar, entry, false)
		}
		fd, err := os.OpenFile(entry.name, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, entry.mode)
		if err != nil {
			log.Fatal(err)
		}
		ar.output(entry, fd)
		fd.Close()
	} else {
		ar.skip(entry)
	}
}
