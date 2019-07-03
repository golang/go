// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package version implements the ``go version'' command.
package version

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"cmd/go/internal/base"
)

var CmdVersion = &base.Command{
	UsageLine: "go version [-m] [-v] [file ...]",
	Short:     "print Go version",
	Long: `Version prints the build information for Go executables.

Go version reports the Go version used to build each of the named
executable files.

If no files are named on the command line, go version prints its own
version information.

If a directory is named, go version walks that directory, recursively,
looking for recognized Go binaries and reporting their versions.
By default, go version does not report unrecognized files found
during a directory scan. The -v flag causes it to report unrecognized files.

The -m flag causes go version to print each executable's embedded
module version information, when available. In the output, the module
information consists of multiple lines following the version line, each
indented by a leading tab character.

See also: go doc runtime/debug.BuildInfo.
`,
}

func init() {
	CmdVersion.Run = runVersion // break init cycle
}

var (
	versionM = CmdVersion.Flag.Bool("m", false, "")
	versionV = CmdVersion.Flag.Bool("v", false, "")
)

func runVersion(cmd *base.Command, args []string) {
	if len(args) == 0 {
		fmt.Printf("go version %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
		return
	}

	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			continue
		}
		if info.IsDir() {
			scanDir(arg)
		} else {
			scanFile(arg, info, true)
		}
	}
}

// scanDir scans a directory for executables to run scanFile on.
func scanDir(dir string) {
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
			scanFile(path, info, *versionV)
		}
		return nil
	})
}

// isExe reports whether the file should be considered executable.
func isExe(file string, info os.FileInfo) bool {
	if runtime.GOOS == "windows" {
		return strings.HasSuffix(strings.ToLower(file), ".exe")
	}
	return info.Mode().IsRegular() && info.Mode()&0111 != 0
}

// scanFile scans file to try to report the Go and module versions.
// If mustPrint is true, scanFile will report any error reading file.
// Otherwise (mustPrint is false, because scanFile is being called
// by scanDir) scanFile prints nothing for non-Go executables.
func scanFile(file string, info os.FileInfo, mustPrint bool) {
	if info.Mode()&os.ModeSymlink != 0 {
		// Accept file symlinks only.
		i, err := os.Stat(file)
		if err != nil || !i.Mode().IsRegular() {
			if mustPrint {
				fmt.Fprintf(os.Stderr, "%s: symlink\n", file)
			}
			return
		}
		info = i
	}
	if !isExe(file, info) {
		if mustPrint {
			fmt.Fprintf(os.Stderr, "%s: not executable file\n", file)
		}
		return
	}

	x, err := openExe(file)
	if err != nil {
		if mustPrint {
			fmt.Fprintf(os.Stderr, "%s: %v\n", file, err)
		}
		return
	}
	defer x.Close()

	vers, mod := findVers(x)
	if vers == "" {
		if mustPrint {
			fmt.Fprintf(os.Stderr, "%s: go version not found\n", file)
		}
		return
	}

	fmt.Printf("%s: %s\n", file, vers)
	if *versionM && mod != "" {
		fmt.Printf("\t%s\n", strings.Replace(mod[:len(mod)-1], "\n", "\n\t", -1))
	}
}

// The build info blob left by the linker is identified by
// a 16-byte header, consisting of buildInfoMagic (14 bytes),
// the binary's pointer size (1 byte),
// and whether the binary is big endian (1 byte).
var buildInfoMagic = []byte("\xff Go buildinf:")

// findVers finds and returns the Go version and module version information
// in the executable x.
func findVers(x exe) (vers, mod string) {
	// Read the first 64kB of text to find the build info blob.
	text := x.DataStart()
	data, err := x.ReadData(text, 64*1024)
	if err != nil {
		return
	}
	for ; !bytes.HasPrefix(data, buildInfoMagic); data = data[32:] {
		if len(data) < 32 {
			return
		}
	}

	// Decode the blob.
	ptrSize := int(data[14])
	bigEndian := data[15] != 0
	var bo binary.ByteOrder
	if bigEndian {
		bo = binary.BigEndian
	} else {
		bo = binary.LittleEndian
	}
	var readPtr func([]byte) uint64
	if ptrSize == 4 {
		readPtr = func(b []byte) uint64 { return uint64(bo.Uint32(b)) }
	} else {
		readPtr = bo.Uint64
	}
	vers = readString(x, ptrSize, readPtr, readPtr(data[16:]))
	if vers == "" {
		return
	}
	mod = readString(x, ptrSize, readPtr, readPtr(data[16+ptrSize:]))
	if len(mod) >= 33 && mod[len(mod)-17] == '\n' {
		// Strip module framing.
		mod = mod[16 : len(mod)-16]
	} else {
		mod = ""
	}
	return
}

// readString returns the string at address addr in the executable x.
func readString(x exe, ptrSize int, readPtr func([]byte) uint64, addr uint64) string {
	hdr, err := x.ReadData(addr, uint64(2*ptrSize))
	if err != nil || len(hdr) < 2*ptrSize {
		return ""
	}
	dataAddr := readPtr(hdr)
	dataLen := readPtr(hdr[ptrSize:])
	data, err := x.ReadData(dataAddr, dataLen)
	if err != nil || uint64(len(data)) < dataLen {
		return ""
	}
	return string(data)
}
