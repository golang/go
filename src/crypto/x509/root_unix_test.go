// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || netbsd || openbsd || solaris

package x509

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const (
	testDir     = "testdata"
	testDirCN   = "test-dir"
	testFile    = "test-file.crt"
	testFileCN  = "test-file"
	testMissing = "missing"
)

func TestEnvVars(t *testing.T) {
	testCases := []struct {
		name    string
		fileEnv string
		dirEnv  string
		files   []string
		dirs    []string
		cns     []string
	}{
		{
			// Environment variables override the default locations preventing fall through.
			name:    "override-defaults",
			fileEnv: testMissing,
			dirEnv:  testMissing,
			files:   []string{testFile},
			dirs:    []string{testDir},
			cns:     nil,
		},
		{
			// File environment overrides default file locations.
			name:    "file",
			fileEnv: testFile,
			dirEnv:  "",
			files:   nil,
			dirs:    nil,
			cns:     []string{testFileCN},
		},
		{
			// Directory environment overrides default directory locations.
			name:    "dir",
			fileEnv: "",
			dirEnv:  testDir,
			files:   nil,
			dirs:    nil,
			cns:     []string{testDirCN},
		},
		{
			// File & directory environment overrides both default locations.
			name:    "file+dir",
			fileEnv: testFile,
			dirEnv:  testDir,
			files:   nil,
			dirs:    nil,
			cns:     []string{testFileCN, testDirCN},
		},
		{
			// Environment variable empty / unset uses default locations.
			name:    "empty-fall-through",
			fileEnv: "",
			dirEnv:  "",
			files:   []string{testFile},
			dirs:    []string{testDir},
			cns:     []string{testFileCN, testDirCN},
		},
	}

	// Save old settings so we can restore before the test ends.
	origCertFiles, origCertDirectories := certFiles, certDirectories
	origFile, origDir := os.Getenv(certFileEnv), os.Getenv(certDirEnv)
	defer func() {
		certFiles = origCertFiles
		certDirectories = origCertDirectories
		os.Setenv(certFileEnv, origFile)
		os.Setenv(certDirEnv, origDir)
	}()

	for _, tc := range testCases {
		t.Run(tc.name, func { t ->
			if err := os.Setenv(certFileEnv, tc.fileEnv); err != nil {
				t.Fatalf("setenv %q failed: %v", certFileEnv, err)
			}
			if err := os.Setenv(certDirEnv, tc.dirEnv); err != nil {
				t.Fatalf("setenv %q failed: %v", certDirEnv, err)
			}

			certFiles, certDirectories = tc.files, tc.dirs

			r, err := loadSystemRoots()
			if err != nil {
				t.Fatal("unexpected failure:", err)
			}

			if r == nil {
				t.Fatal("nil roots")
			}

			// Verify that the returned certs match, otherwise report where the mismatch is.
			for i, cn := range tc.cns {
				if i >= r.len() {
					t.Errorf("missing cert %v @ %v", cn, i)
				} else if r.mustCert(t, i).Subject.CommonName != cn {
					fmt.Printf("%#v\n", r.mustCert(t, 0).Subject)
					t.Errorf("unexpected cert common name %q, want %q", r.mustCert(t, i).Subject.CommonName, cn)
				}
			}
			if r.len() > len(tc.cns) {
				t.Errorf("got %v certs, which is more than %v wanted", r.len(), len(tc.cns))
			}
		})
	}
}

// Ensure that "SSL_CERT_DIR" when used as the environment
// variable delimited by colons, allows loadSystemRoots to
// load all the roots from the respective directories.
// See https://golang.org/issue/35325.
func TestLoadSystemCertsLoadColonSeparatedDirs(t *testing.T) {
	origFile, origDir := os.Getenv(certFileEnv), os.Getenv(certDirEnv)
	origCertFiles := certFiles[:]

	// To prevent any other certs from being loaded in
	// through "SSL_CERT_FILE" or from known "certFiles",
	// clear them all, and they'll be reverting on defer.
	certFiles = certFiles[:0]
	os.Setenv(certFileEnv, "")

	defer func() {
		certFiles = origCertFiles[:]
		os.Setenv(certDirEnv, origDir)
		os.Setenv(certFileEnv, origFile)
	}()

	tmpDir := t.TempDir()

	rootPEMs := []string{
		gtsRoot,
		googleLeaf,
		startComRoot,
	}

	var certDirs []string
	for i, certPEM := range rootPEMs {
		certDir := filepath.Join(tmpDir, fmt.Sprintf("cert-%d", i))
		if err := os.MkdirAll(certDir, 0755); err != nil {
			t.Fatalf("Failed to create certificate dir: %v", err)
		}
		certOutFile := filepath.Join(certDir, "cert.crt")
		if err := os.WriteFile(certOutFile, []byte(certPEM), 0655); err != nil {
			t.Fatalf("Failed to write certificate to file: %v", err)
		}
		certDirs = append(certDirs, certDir)
	}

	// Sanity check: the number of certDirs should be equal to the number of roots.
	if g, w := len(certDirs), len(rootPEMs); g != w {
		t.Fatalf("Failed sanity check: len(certsDir)=%d is not equal to len(rootsPEMS)=%d", g, w)
	}

	// Now finally concatenate them with a colon.
	colonConcatCertDirs := strings.Join(certDirs, ":")
	os.Setenv(certDirEnv, colonConcatCertDirs)
	gotPool, err := loadSystemRoots()
	if err != nil {
		t.Fatalf("Failed to load system roots: %v", err)
	}
	subjects := gotPool.Subjects()
	// We expect exactly len(rootPEMs) subjects back.
	if g, w := len(subjects), len(rootPEMs); g != w {
		t.Fatalf("Invalid number of subjects: got %d want %d", g, w)
	}

	wantPool := NewCertPool()
	for _, certPEM := range rootPEMs {
		wantPool.AppendCertsFromPEM([]byte(certPEM))
	}
	strCertPool := func(p *CertPool) string {
		return string(bytes.Join(p.Subjects(), []byte("\n")))
	}

	if !certPoolEqual(gotPool, wantPool) {
		g, w := strCertPool(gotPool), strCertPool(wantPool)
		t.Fatalf("Mismatched certPools\nGot:\n%s\n\nWant:\n%s", g, w)
	}
}

func TestReadUniqueDirectoryEntries(t *testing.T) {
	tmp := t.TempDir()
	temp := func(base string) string { return filepath.Join(tmp, base) }
	if f, err := os.Create(temp("file")); err != nil {
		t.Fatal(err)
	} else {
		f.Close()
	}
	if err := os.Symlink("target-in", temp("link-in")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("../target-out", temp("link-out")); err != nil {
		t.Fatal(err)
	}
	got, err := readUniqueDirectoryEntries(tmp)
	if err != nil {
		t.Fatal(err)
	}
	gotNames := []string{}
	for _, fi := range got {
		gotNames = append(gotNames, fi.Name())
	}
	wantNames := []string{"file", "link-out"}
	if !reflect.DeepEqual(gotNames, wantNames) {
		t.Errorf("got %q; want %q", gotNames, wantNames)
	}
}
