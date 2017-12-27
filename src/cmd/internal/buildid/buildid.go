// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildid

import (
	"bytes"
	"debug/elf"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

var (
	errBuildIDToolchain = fmt.Errorf("build ID only supported in gc toolchain")
	errBuildIDMalformed = fmt.Errorf("malformed object file")
	errBuildIDUnknown   = fmt.Errorf("lost build ID")
)

var (
	bangArch = []byte("!<arch>")
	pkgdef   = []byte("__.PKGDEF")
	goobject = []byte("go object ")
	buildid  = []byte("build id ")
)

// ReadFile reads the build ID from an archive or executable file.
func ReadFile(name string) (id string, err error) {
	f, err := os.Open(name)
	if err != nil {
		return "", err
	}
	defer f.Close()

	buf := make([]byte, 8)
	if _, err := f.ReadAt(buf, 0); err != nil {
		return "", err
	}
	if string(buf) != "!<arch>\n" {
		return readBinary(name, f)
	}

	// Read just enough of the target to fetch the build ID.
	// The archive is expected to look like:
	//
	//	!<arch>
	//	__.PKGDEF       0           0     0     644     7955      `
	//	go object darwin amd64 devel X:none
	//	build id "b41e5c45250e25c9fd5e9f9a1de7857ea0d41224"
	//
	// The variable-sized strings are GOOS, GOARCH, and the experiment list (X:none).
	// Reading the first 1024 bytes should be plenty.
	data := make([]byte, 1024)
	n, err := io.ReadFull(f, data)
	if err != nil && n == 0 {
		return "", err
	}

	tryGccgo := func() (string, error) {
		return readGccgoArchive(name, f)
	}

	// Archive header.
	for i := 0; ; i++ { // returns during i==3
		j := bytes.IndexByte(data, '\n')
		if j < 0 {
			return tryGccgo()
		}
		line := data[:j]
		data = data[j+1:]
		switch i {
		case 0:
			if !bytes.Equal(line, bangArch) {
				return tryGccgo()
			}
		case 1:
			if !bytes.HasPrefix(line, pkgdef) {
				return tryGccgo()
			}
		case 2:
			if !bytes.HasPrefix(line, goobject) {
				return tryGccgo()
			}
		case 3:
			if !bytes.HasPrefix(line, buildid) {
				// Found the object header, just doesn't have a build id line.
				// Treat as successful, with empty build id.
				return "", nil
			}
			id, err := strconv.Unquote(string(line[len(buildid):]))
			if err != nil {
				return tryGccgo()
			}
			return id, nil
		}
	}
}

// readGccgoArchive tries to parse the archive as a standard Unix
// archive file, and fetch the build ID from the _buildid.o entry.
// The _buildid.o entry is written by (*Builder).gccgoBuildIDELFFile
// in cmd/go/internal/work/exec.go.
func readGccgoArchive(name string, f *os.File) (string, error) {
	bad := func() (string, error) {
		return "", &os.PathError{Op: "parse", Path: name, Err: errBuildIDMalformed}
	}

	off := int64(8)
	for {
		if _, err := f.Seek(off, io.SeekStart); err != nil {
			return "", err
		}

		// TODO(iant): Make a debug/ar package, and use it
		// here and in cmd/link.
		var hdr [60]byte
		if _, err := io.ReadFull(f, hdr[:]); err != nil {
			if err == io.EOF {
				// No more entries, no build ID.
				return "", nil
			}
			return "", err
		}
		off += 60

		sizeStr := strings.TrimSpace(string(hdr[48:58]))
		size, err := strconv.ParseInt(sizeStr, 0, 64)
		if err != nil {
			return bad()
		}

		name := strings.TrimSpace(string(hdr[:16]))
		if name == "_buildid.o/" {
			sr := io.NewSectionReader(f, off, size)
			e, err := elf.NewFile(sr)
			if err != nil {
				return bad()
			}
			s := e.Section(".go.buildid")
			if s == nil {
				return bad()
			}
			data, err := s.Data()
			if err != nil {
				return bad()
			}
			return string(data), nil
		}

		off += size
		if off&1 != 0 {
			off++
		}
	}
}

var (
	goBuildPrefix = []byte("\xff Go build ID: \"")
	goBuildEnd    = []byte("\"\n \xff")

	elfPrefix = []byte("\x7fELF")

	machoPrefixes = [][]byte{
		{0xfe, 0xed, 0xfa, 0xce},
		{0xfe, 0xed, 0xfa, 0xcf},
		{0xce, 0xfa, 0xed, 0xfe},
		{0xcf, 0xfa, 0xed, 0xfe},
	}
)

var readSize = 32 * 1024 // changed for testing

// readBinary reads the build ID from a binary.
//
// ELF binaries store the build ID in a proper PT_NOTE section.
//
// Other binary formats are not so flexible. For those, the linker
// stores the build ID as non-instruction bytes at the very beginning
// of the text segment, which should appear near the beginning
// of the file. This is clumsy but fairly portable. Custom locations
// can be added for other binary types as needed, like we did for ELF.
func readBinary(name string, f *os.File) (id string, err error) {
	// Read the first 32 kB of the binary file.
	// That should be enough to find the build ID.
	// In ELF files, the build ID is in the leading headers,
	// which are typically less than 4 kB, not to mention 32 kB.
	// In Mach-O files, there's no limit, so we have to parse the file.
	// On other systems, we're trying to read enough that
	// we get the beginning of the text segment in the read.
	// The offset where the text segment begins in a hello
	// world compiled for each different object format today:
	//
	//	Plan 9: 0x20
	//	Windows: 0x600
	//
	data := make([]byte, readSize)
	_, err = io.ReadFull(f, data)
	if err == io.ErrUnexpectedEOF {
		err = nil
	}
	if err != nil {
		return "", err
	}

	if bytes.HasPrefix(data, elfPrefix) {
		return readELF(name, f, data)
	}
	for _, m := range machoPrefixes {
		if bytes.HasPrefix(data, m) {
			return readMacho(name, f, data)
		}
	}
	return readRaw(name, data)
}

// readRaw finds the raw build ID stored in text segment data.
func readRaw(name string, data []byte) (id string, err error) {
	i := bytes.Index(data, goBuildPrefix)
	if i < 0 {
		// Missing. Treat as successful but build ID empty.
		return "", nil
	}

	j := bytes.Index(data[i+len(goBuildPrefix):], goBuildEnd)
	if j < 0 {
		return "", &os.PathError{Op: "parse", Path: name, Err: errBuildIDMalformed}
	}

	quoted := data[i+len(goBuildPrefix)-1 : i+len(goBuildPrefix)+j+1]
	id, err = strconv.Unquote(string(quoted))
	if err != nil {
		return "", &os.PathError{Op: "parse", Path: name, Err: errBuildIDMalformed}
	}
	return id, nil
}
