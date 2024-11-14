// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildid

import (
	"bytes"
	"debug/elf"
	"fmt"
	"internal/xcoff"
	"io"
	"io/fs"
	"os"
	"strconv"
	"strings"
)

var (
	errBuildIDMalformed = fmt.Errorf("malformed object file")

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
		if string(buf) == "<bigaf>\n" {
			return readGccgoBigArchive(name, f)
		}
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

	tryGccgo := func { readGccgoArchive(name, f) }

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
	bad := func { "", &fs.PathError{Op: "parse", Path: name, Err: errBuildIDMalformed} }

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

// readGccgoBigArchive tries to parse the archive as an AIX big
// archive file, and fetch the build ID from the _buildid.o entry.
// The _buildid.o entry is written by (*Builder).gccgoBuildIDXCOFFFile
// in cmd/go/internal/work/exec.go.
func readGccgoBigArchive(name string, f *os.File) (string, error) {
	bad := func { "", &fs.PathError{Op: "parse", Path: name, Err: errBuildIDMalformed} }

	// Read fixed-length header.
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	var flhdr [128]byte
	if _, err := io.ReadFull(f, flhdr[:]); err != nil {
		return "", err
	}
	// Read first member offset.
	offStr := strings.TrimSpace(string(flhdr[68:88]))
	off, err := strconv.ParseInt(offStr, 10, 64)
	if err != nil {
		return bad()
	}
	for {
		if off == 0 {
			// No more entries, no build ID.
			return "", nil
		}
		if _, err := f.Seek(off, io.SeekStart); err != nil {
			return "", err
		}
		// Read member header.
		var hdr [112]byte
		if _, err := io.ReadFull(f, hdr[:]); err != nil {
			return "", err
		}
		// Read member name length.
		namLenStr := strings.TrimSpace(string(hdr[108:112]))
		namLen, err := strconv.ParseInt(namLenStr, 10, 32)
		if err != nil {
			return bad()
		}
		if namLen == 10 {
			var nam [10]byte
			if _, err := io.ReadFull(f, nam[:]); err != nil {
				return "", err
			}
			if string(nam[:]) == "_buildid.o" {
				sizeStr := strings.TrimSpace(string(hdr[0:20]))
				size, err := strconv.ParseInt(sizeStr, 10, 64)
				if err != nil {
					return bad()
				}
				off += int64(len(hdr)) + namLen + 2
				if off&1 != 0 {
					off++
				}
				sr := io.NewSectionReader(f, off, size)
				x, err := xcoff.NewFile(sr)
				if err != nil {
					return bad()
				}
				data := x.CSect(".go.buildid")
				if data == nil {
					return bad()
				}
				return string(data), nil
			}
		}

		// Read next member offset.
		offStr = strings.TrimSpace(string(hdr[20:40]))
		off, err = strconv.ParseInt(offStr, 10, 64)
		if err != nil {
			return bad()
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
		return "", &fs.PathError{Op: "parse", Path: name, Err: errBuildIDMalformed}
	}

	quoted := data[i+len(goBuildPrefix)-1 : i+len(goBuildPrefix)+j+1]
	id, err = strconv.Unquote(string(quoted))
	if err != nil {
		return "", &fs.PathError{Op: "parse", Path: name, Err: errBuildIDMalformed}
	}
	return id, nil
}

// HashToString converts the hash h to a string to be recorded
// in package archives and binaries as part of the build ID.
// We use the first 120 bits of the hash (5 chunks of 24 bits each) and encode
// it in base64, resulting in a 20-byte string. Because this is only used for
// detecting the need to rebuild installed files (not for lookups
// in the object file cache), 120 bits are sufficient to drive the
// probability of a false "do not need to rebuild" decision to effectively zero.
// We embed two different hashes in archives and four in binaries,
// so cutting to 20 bytes is a significant savings when build IDs are displayed.
// (20*4+3 = 83 bytes compared to 64*4+3 = 259 bytes for the
// more straightforward option of printing the entire h in base64).
func HashToString(h [32]byte) string {
	const b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
	const chunks = 5
	var dst [chunks * 4]byte
	for i := 0; i < chunks; i++ {
		v := uint32(h[3*i])<<16 | uint32(h[3*i+1])<<8 | uint32(h[3*i+2])
		dst[4*i+0] = b64[(v>>18)&0x3F]
		dst[4*i+1] = b64[(v>>12)&0x3F]
		dst[4*i+2] = b64[(v>>6)&0x3F]
		dst[4*i+3] = b64[v&0x3F]
	}
	return string(dst[:])
}
