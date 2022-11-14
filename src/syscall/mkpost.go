// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// mkpost processes the output of cgo -godefs to
// modify the generated types. It is used to clean up
// the syscall API in an architecture specific manner.
//
// mkpost is run after cgo -godefs by mkall.sh.
package main

import (
	"fmt"
	"go/format"
	"io"
	"log"
	"os"
	"regexp"
	"strings"
)

func main() {
	b, err := io.ReadAll(os.Stdin)
	if err != nil {
		log.Fatal(err)
	}
	s := string(b)

	goarch := os.Getenv("GOARCH")
	goos := os.Getenv("GOOS")
	switch {
	case goarch == "s390x" && goos == "linux":
		// Export the types of PtraceRegs fields.
		re := regexp.MustCompile("ptrace(Psw|Fpregs|Per)")
		s = re.ReplaceAllString(s, "Ptrace$1")

		// Replace padding fields inserted by cgo with blank identifiers.
		re = regexp.MustCompile("Pad_cgo[A-Za-z0-9_]*")
		s = re.ReplaceAllString(s, "_")

		// We want to keep X__val in Fsid. Hide it and restore it later.
		s = strings.Replace(s, "X__val", "MKPOSTFSIDVAL", 1)

		// Replace other unwanted fields with blank identifiers.
		re = regexp.MustCompile("X_[A-Za-z0-9_]*")
		s = re.ReplaceAllString(s, "_")

		// Restore X__val in Fsid.
		s = strings.Replace(s, "MKPOSTFSIDVAL", "X__val", 1)

		// Force the type of RawSockaddr.Data to [14]int8 to match
		// the existing gccgo API.
		re = regexp.MustCompile("(Data\\s+\\[14\\])uint8")
		s = re.ReplaceAllString(s, "${1}int8")

	case goos == "freebsd":
		// Keep pre-FreeBSD 10 / non-POSIX 2008 names for timespec fields
		re := regexp.MustCompile("(A|M|C|Birth)tim\\s+Timespec")
		s = re.ReplaceAllString(s, "${1}timespec Timespec")
	}

	// gofmt
	b, err = format.Source([]byte(s))
	if err != nil {
		log.Fatal(err)
	}

	// Append this command to the header to show where the new file
	// came from.
	re := regexp.MustCompile("(cgo -godefs [a-zA-Z0-9_]+\\.go.*)")
	s = re.ReplaceAllString(string(b), "$1 | go run mkpost.go")

	fmt.Print(s)
}
