// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildid

import (
	"bytes"
	"cmd/go/internal/cfg"
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

// ReadBuildID reads the build ID from an archive or binary.
// It only supports the gc toolchain.
// Other toolchain maintainers should adjust this function.
func ReadBuildID(name, target string) (id string, err error) {
	if cfg.BuildToolchainName != "gc" {
		return "", errBuildIDToolchain
	}

	// For commands, read build ID directly from binary.
	if name == "main" {
		return ReadBuildIDFromBinary(target)
	}

	// Otherwise, we expect to have an archive (.a) file,
	// and we can read the build ID from the Go export data.
	if !strings.HasSuffix(target, ".a") {
		return "", &os.PathError{Op: "parse", Path: target, Err: errBuildIDUnknown}
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
	f, err := os.Open(target)
	if err != nil {
		return "", err
	}
	data := make([]byte, 1024)
	n, err := io.ReadFull(f, data)
	f.Close()

	if err != nil && n == 0 {
		return "", err
	}

	bad := func() (string, error) {
		return "", &os.PathError{Op: "parse", Path: target, Err: errBuildIDMalformed}
	}

	// Archive header.
	for i := 0; ; i++ { // returns during i==3
		j := bytes.IndexByte(data, '\n')
		if j < 0 {
			return bad()
		}
		line := data[:j]
		data = data[j+1:]
		switch i {
		case 0:
			if !bytes.Equal(line, bangArch) {
				return bad()
			}
		case 1:
			if !bytes.HasPrefix(line, pkgdef) {
				return bad()
			}
		case 2:
			if !bytes.HasPrefix(line, goobject) {
				return bad()
			}
		case 3:
			if !bytes.HasPrefix(line, buildid) {
				// Found the object header, just doesn't have a build id line.
				// Treat as successful, with empty build id.
				return "", nil
			}
			id, err := strconv.Unquote(string(line[len(buildid):]))
			if err != nil {
				return bad()
			}
			return id, nil
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

var BuildIDReadSize = 32 * 1024 // changed for testing

// ReadBuildIDFromBinary reads the build ID from a binary.
//
// ELF binaries store the build ID in a proper PT_NOTE section.
//
// Other binary formats are not so flexible. For those, the linker
// stores the build ID as non-instruction bytes at the very beginning
// of the text segment, which should appear near the beginning
// of the file. This is clumsy but fairly portable. Custom locations
// can be added for other binary types as needed, like we did for ELF.
func ReadBuildIDFromBinary(filename string) (id string, err error) {
	if filename == "" {
		return "", &os.PathError{Op: "parse", Path: filename, Err: errBuildIDUnknown}
	}

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
	f, err := os.Open(filename)
	if err != nil {
		return "", err
	}
	defer f.Close()

	data := make([]byte, BuildIDReadSize)
	_, err = io.ReadFull(f, data)
	if err == io.ErrUnexpectedEOF {
		err = nil
	}
	if err != nil {
		return "", err
	}

	if bytes.HasPrefix(data, elfPrefix) {
		return readELFGoBuildID(filename, f, data)
	}
	for _, m := range machoPrefixes {
		if bytes.HasPrefix(data, m) {
			return readMachoGoBuildID(filename, f, data)
		}
	}

	return readRawGoBuildID(filename, data)
}

// readRawGoBuildID finds the raw build ID stored in text segment data.
func readRawGoBuildID(filename string, data []byte) (id string, err error) {
	i := bytes.Index(data, goBuildPrefix)
	if i < 0 {
		// Missing. Treat as successful but build ID empty.
		return "", nil
	}

	j := bytes.Index(data[i+len(goBuildPrefix):], goBuildEnd)
	if j < 0 {
		return "", &os.PathError{Op: "parse", Path: filename, Err: errBuildIDMalformed}
	}

	quoted := data[i+len(goBuildPrefix)-1 : i+len(goBuildPrefix)+j+1]
	id, err = strconv.Unquote(string(quoted))
	if err != nil {
		return "", &os.PathError{Op: "parse", Path: filename, Err: errBuildIDMalformed}
	}

	return id, nil
}
