// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mkzip creates a FIPS snapshot zip file.
// See GOROOT/lib/fips140/README.md and GOROOT/lib/fips140/Makefile
// for more details about when and why to use this.
//
// Usage:
//
//	cd GOROOT/src/cmd/go/internal/fips140/_mkzip
//	GOTOOLCHAIN=go1.26.10 go run . [-b branch] v1.2.3
//
// Mkzip creates a zip file named for the version on the command line
// using the sources in the named branch (default origin/master,
// to avoid accidentally including local commits).
//
// Mkzip must be run with Go 1.26, so that all snapshot zip files are
// compressed by the same compress/flate encoder and can be stored by
// git as small deltas against previous ones.
package main

import (
	"archive/zip"
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
	modzip "golang.org/x/mod/zip"
)

var flagBranch = flag.String("b", "origin/master", "branch to use")

func usage() {
	fmt.Fprintf(os.Stderr, "usage: GOTOOLCHAIN=go1.26.10 go run . [-b branch] vX.Y.Z\n")
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("mkzip: ")
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 1 {
		usage()
	}

	if !strings.HasPrefix(runtime.Version(), "go1.26.") {
		log.Fatalf("must be run with GOTOOLCHAIN=go1.26.10, not %s", runtime.Version())
	}

	// Must run in this module's directory. The snapshots live in lib/fips140.
	goroot := "../../../../../.."
	dir := filepath.Join(goroot, "lib/fips140")
	if _, err := os.Stat(filepath.Join(dir, "fips140.sum")); err != nil {
		log.Fatalf("must be run in the src/cmd/go/internal/fips140/_mkzip directory")
	}

	// Must have valid version, and must not overwrite existing file.
	version := flag.Arg(0)
	if semver.Canonical(version) != version {
		log.Fatalf("invalid version %q; must be vX.Y.Z", version)
	}
	zipFile := filepath.Join(dir, version+".zip")
	if _, err := os.Stat(zipFile); err == nil {
		log.Fatalf("%s.zip already exists", version)
	}

	// Make standard module zip file in memory.
	// The module path "golang.org/fips140" needs to be a valid module name,
	// and it is the path where the zip file will be unpacked in the module cache.
	// The path must begin with a domain name to satisfy the module validation rules,
	// but otherwise the path is not used. The cmd/go code using these zips
	// knows that the zip contains crypto/internal/fips140.
	var zbuf bytes.Buffer
	err := modzip.CreateFromVCS(&zbuf,
		module.Version{Path: "golang.org/fips140", Version: version},
		goroot, *flagBranch, "src/crypto/internal/fips140")
	if err != nil {
		log.Fatal(err)
	}

	// Write new zip file with longer paths: fips140/v1.2.3/foo.go instead of foo.go.
	// That way we can bind the fips140 directory onto the
	// GOROOT/src/crypto/internal/fips140 directory and get a
	// crypto/internal/fips140/v1.2.3 with the snapshot code
	// and an otherwise empty crypto/internal/fips140 directory.
	zr, err := zip.NewReader(bytes.NewReader(zbuf.Bytes()), int64(zbuf.Len()))
	if err != nil {
		log.Fatal(err)
	}

	var zbuf2 bytes.Buffer
	zw := zip.NewWriter(&zbuf2)
	foundVersion := false
	for _, f := range zr.File {
		// golang.org/fips140@v1.2.3/dir/file.go ->
		// golang.org/fips140@v1.2.3/fips140/v1.2.3/dir/file.go
		if f.Name != "golang.org/fips140@"+version+"/LICENSE" {
			f.Name = "golang.org/fips140@" + version + "/fips140/" + version +
				strings.TrimPrefix(f.Name, "golang.org/fips140@"+version)
		}
		// Inject version in [crypto/internal/fips140.Version].
		if f.Name == "golang.org/fips140@"+version+"/fips140/"+version+"/fips140.go" {
			rf, err := f.Open()
			if err != nil {
				log.Fatal(err)
			}
			contents, err := io.ReadAll(rf)
			if err != nil {
				log.Fatal(err)
			}
			returnLine := `return "latest" //mkzip:version`
			if !bytes.Contains(contents, []byte(returnLine)) {
				log.Fatalf("did not find %q in fips140.go", returnLine)
			}
			// Use only the vX.Y.Z part of a possible vX.Y.Z-hash version.
			v, _, _ := strings.Cut(version, "-")
			newLine := `return "` + v + `"`
			contents = bytes.ReplaceAll(contents, []byte(returnLine), []byte(newLine))
			wf, err := zw.Create(f.Name)
			if err != nil {
				log.Fatal(err)
			}
			if _, err := wf.Write(contents); err != nil {
				log.Fatal(err)
			}
			foundVersion = true
			continue
		}
		wf, err := zw.CreateRaw(&f.FileHeader)
		if err != nil {
			log.Fatal(err)
		}
		rf, err := f.OpenRaw()
		if err != nil {
			log.Fatal(err)
		}
		if _, err := io.Copy(wf, rf); err != nil {
			log.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		log.Fatal(err)
	}
	if !foundVersion {
		log.Fatal("did not find fips140.go file")
	}

	err = os.WriteFile(zipFile, zbuf2.Bytes(), 0666)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("wrote %s.zip", version)
}
