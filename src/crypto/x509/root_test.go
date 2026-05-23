// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestFallbackPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Multiple calls to SetFallbackRoots should panic")
		}
	}()
	SetFallbackRoots(nil)
	SetFallbackRoots(nil)
}

func TestFallback(t *testing.T) {
	// call systemRootsPool so that the sync.Once is triggered, and we can
	// manipulate systemRoots without worrying about our working being overwritten
	systemRootsPool()
	if systemRoots != nil {
		originalSystemRoots := *systemRoots
		defer func() { systemRoots = &originalSystemRoots }()
	}

	tests := []struct {
		name            string
		systemRoots     *CertPool
		systemPool      bool
		poolContent     []*Certificate
		forceFallback   bool
		returnsFallback bool
	}{
		{
			name:            "nil systemRoots",
			returnsFallback: true,
		},
		{
			name:            "empty systemRoots",
			systemRoots:     NewCertPool(),
			returnsFallback: true,
		},
		{
			name:        "empty systemRoots system pool",
			systemRoots: NewCertPool(),
			systemPool:  true,
		},
		{
			name:        "filled systemRoots system pool",
			systemRoots: NewCertPool(),
			poolContent: []*Certificate{{}},
			systemPool:  true,
		},
		{
			name:        "filled systemRoots",
			systemRoots: NewCertPool(),
			poolContent: []*Certificate{{}},
		},
		{
			name:            "filled systemRoots, force fallback",
			systemRoots:     NewCertPool(),
			poolContent:     []*Certificate{{}},
			forceFallback:   true,
			returnsFallback: true,
		},
		{
			name:            "filled systemRoot system pool, force fallback",
			systemRoots:     NewCertPool(),
			poolContent:     []*Certificate{{}},
			systemPool:      true,
			forceFallback:   true,
			returnsFallback: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			useFallbackRoots = false
			fallbacksSet = false
			systemRoots = tc.systemRoots

			if systemRoots != nil {
				systemRoots.systemPool = tc.systemPool
			}
			for _, c := range tc.poolContent {
				systemRoots.AddCert(c)
			}
			if tc.forceFallback {
				t.Setenv("GODEBUG", "x509usefallbackroots=1")
			} else {
				t.Setenv("GODEBUG", "x509usefallbackroots=0")
			}

			fallbackPool := NewCertPool()
			SetFallbackRoots(fallbackPool)

			systemPoolIsFallback := systemRoots == fallbackPool

			if tc.returnsFallback && !systemPoolIsFallback {
				t.Error("systemRoots was not set to fallback pool")
			} else if !tc.returnsFallback && systemPoolIsFallback {
				t.Error("systemRoots was set to fallback pool when it shouldn't have been")
			}
		})
	}
}

const (
	testDirCN   = "test-dir"
	testFile    = "test-file.crt"
	testFileCN  = "test-file"
	testMissing = "missing"
)

func TestEnvVars(t *testing.T) {
	tmpDir := t.TempDir()
	testCert, err := os.ReadFile("testdata/test-dir.crt")
	if err != nil {
		t.Fatalf("failed to read test cert: %s", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, testFile), testCert, 0644); err != nil {
		t.Fatalf("failed to write test cert: %s", err)
	}

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
			dirs:    []string{tmpDir},
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
			dirEnv:  tmpDir,
			files:   nil,
			dirs:    nil,
			cns:     []string{testDirCN},
		},
		{
			// File & directory environment overrides both default locations.
			name:    "file+dir",
			fileEnv: testFile,
			dirEnv:  tmpDir,
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
			dirs:    []string{tmpDir},
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
		t.Run(tc.name, func(t *testing.T) {
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

			wantSystemPool := (runtime.GOOS == "darwin" || runtime.GOOS == "windows") && tc.dirEnv == "" && tc.fileEnv == ""

			if wantSystemPool {
				if !r.systemPool {
					t.Fatal("expected returned cert pool to be a system pool")
				}
				if r.len() != 0 {
					t.Fatalf("expected empty system pool, pool has %d roots", r.len())
				}
				return
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

// Ensure that "SSL_CERT_DIR" when used as the environment variable delimited by
// colons on Unix-like systems, and semicolons on Windows, allows
// loadSystemRoots to load all the roots from the respective directories.
// See https://golang.org/issue/35325.
func TestLoadSystemCertsLoadColonSeparatedDirs(t *testing.T) {
	origFile, origDir := os.Getenv(certFileEnv), os.Getenv(certDirEnv)
	origCertFiles := certFiles[:]

	// To prevent any other certs from being loaded in
	// through "SSL_CERT_FILE" or from known "certFiles",
	// clear them all, and they'll be reverted on defer.
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
	}

	var certDirs []string
	for i, certPEM := range rootPEMs {
		certDir := filepath.Join(tmpDir, fmt.Sprintf("cert-%d", i))
		if err := os.MkdirAll(certDir, 0755); err != nil {
			t.Fatalf("failed to create certificate dir: %v", err)
		}
		certOutFile := filepath.Join(certDir, "cert.crt")
		if err := os.WriteFile(certOutFile, []byte(certPEM), 0655); err != nil {
			t.Fatalf("failed to write certificate to file: %v", err)
		}
		certDirs = append(certDirs, certDir)
	}

	// Sanity check: the number of certDirs should be equal to the number of roots.
	if g, w := len(certDirs), len(rootPEMs); g != w {
		t.Fatalf("failed sanity check: len(certsDir)=%d is not equal to len(rootsPEMS)=%d", g, w)
	}

	// Now finally concatenate them with a colon/semicolon.
	concatCertDirs := strings.Join(certDirs, string(filepath.ListSeparator))
	os.Setenv(certDirEnv, concatCertDirs)
	gotPool, err := loadSystemRoots()
	if err != nil {
		t.Fatalf("failed to load system roots: %v", err)
	}
	subjects := gotPool.Subjects()
	// We expect exactly len(rootPEMs) subjects back.
	if g, w := len(subjects), len(rootPEMs); g != w {
		t.Fatalf("invalid number of subjects: got %d want %d", g, w)
	}

	wantPool := NewCertPool()
	for _, certPEM := range rootPEMs {
		wantPool.AppendCertsFromPEM([]byte(certPEM))
	}
	strCertPool := func(p *CertPool) string {
		return string(bytes.Join(p.Subjects(), []byte("\n")))
	}

	if !certPoolEqual(gotPool, wantPool) {
		got, want := strCertPool(gotPool), strCertPool(wantPool)
		t.Fatalf("mismatched certPools\nGot:\n%s\n\nWant:\n%s", got, want)
	}
}

func TestReadUniqueDirectoryEntries(t *testing.T) {
	testenv.MustHaveSymlink(t)
	baseTmpDir := t.TempDir()
	path := func(base string) string { return filepath.Join(baseTmpDir, base) }
	if f, err := os.Create(path("file")); err != nil {
		t.Fatal(err)
	} else {
		f.Close()
	}
	if err := os.Symlink("target-in", path("link-in")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("../target-out", path("link-out")); err != nil {
		t.Fatal(err)
	}
	got, err := readUniqueDirectoryEntries(baseTmpDir)
	if err != nil {
		t.Fatal(err)
	}
	gotNames := []string{}
	for _, fi := range got {
		gotNames = append(gotNames, fi.Name())
	}
	wantNames := []string{"file", "link-out"}
	if !slices.Equal(gotNames, wantNames) {
		t.Errorf("got %q; want %q", gotNames, wantNames)
	}
}

func TestSSLCertEnvOverride(t *testing.T) {
	testenv.SetGODEBUG(t, "x509sslcertoverrideplatform=0")
	t.Setenv(certFileEnv, "/tmp/nope")
	t.Setenv(certDirEnv, "/tmp/nope")

	p, err := loadSystemRoots()
	if err != nil {
		t.Fatalf("unexpected failure: %s", err)
	}

	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		if !p.systemPool {
			t.Fatal("x509sslcertoverrideplatform did not override SSL_CERT_{FILE,DIR}")
		}
	} else if p.systemPool {
		t.Fatal("x509sslcertoverrideplatform caused a systemPool to be returned on OS other than windows or darwin")
	}
}
