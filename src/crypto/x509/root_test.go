// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"encoding/pem"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
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
		originalSystemRoots := systemRoots
		defer func() { systemRoots = originalSystemRoots }()
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
			// When a bundle file has roots, directories are skipped (see #38869).
			name:    "empty-fall-through",
			fileEnv: "",
			dirEnv:  "",
			files:   []string{testFile},
			dirs:    []string{tmpDir},
			cns:     []string{testFileCN},
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

func TestLoadOnDiskRootsSkipsDirWhenFileHasRoots(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a bundle file with a cert.
	testCert, err := os.ReadFile("testdata/test-dir.crt")
	if err != nil {
		t.Fatalf("failed to read test cert: %s", err)
	}
	bundleFile := filepath.Join(tmpDir, "bundle.crt")
	if err := os.WriteFile(bundleFile, testCert, 0644); err != nil {
		t.Fatal(err)
	}

	// Create a directory with a DISTINCT cert that should NOT be loaded
	// when the bundle already has roots and dirs are not user-provided.
	certDir := filepath.Join(tmpDir, "certs")
	if err := os.MkdirAll(certDir, 0755); err != nil {
		t.Fatal(err)
	}
	dirCA, _, err := generateCert("Distinct Dir CA", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(certDir, "extra.pem"),
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: dirCA.Raw}), 0644); err != nil {
		t.Fatal(err)
	}

	// Save and override defaults.
	origCertFiles, origCertDirectories := certFiles, certDirectories
	defer func() {
		certFiles = origCertFiles
		certDirectories = origCertDirectories
	}()
	certFiles = []string{bundleFile}
	certDirectories = []string{certDir}

	// Load with no user-provided env vars (empty strings = use defaults).
	roots, err := loadOnDiskRoots("", "")
	if err != nil {
		t.Fatal(err)
	}

	// Should have only the certs from the bundle file (1 cert).
	// The directory cert is distinct, so if it were loaded we'd see 2.
	if got := roots.len(); got != 1 {
		t.Errorf("got %d certs, want 1 (directory should be skipped when bundle has roots)", got)
	}
	// Verify directory scan was deferred, not performed eagerly.
	if roots.lazyRoots == nil {
		t.Fatal("expected deferred root state")
	}
	if roots.lazyRoots.pool != nil {
		t.Fatal("directory was scanned eagerly")
	}
}

func TestLoadOnDiskRootsScansDirWhenBundleHasNoRoots(t *testing.T) {
	tmpDir := t.TempDir()

	// Create an empty bundle file (no certs) so it "succeeds" but has no roots.
	bundleFile := filepath.Join(tmpDir, "empty-bundle.crt")
	if err := os.WriteFile(bundleFile, []byte("not a cert"), 0644); err != nil {
		t.Fatal(err)
	}

	// Create a real cert only in the directory.
	certDir := filepath.Join(tmpDir, "certs")
	if err := os.MkdirAll(certDir, 0755); err != nil {
		t.Fatal(err)
	}
	testCert, err := os.ReadFile("testdata/test-dir.crt")
	if err != nil {
		t.Fatalf("failed to read test cert: %s", err)
	}
	if err := os.WriteFile(filepath.Join(certDir, "cert.pem"), testCert, 0644); err != nil {
		t.Fatal(err)
	}

	// Save and override defaults.
	origCertFiles, origCertDirectories := certFiles, certDirectories
	defer func() {
		certFiles = origCertFiles
		certDirectories = origCertDirectories
	}()
	certFiles = []string{bundleFile}
	certDirectories = []string{certDir}

	// Load - bundle has no valid certs, so directories should be scanned eagerly.
	roots, err := loadOnDiskRoots("", "")
	if err != nil {
		t.Fatal(err)
	}

	// Since bundle had zero roots, the directory scan should have happened eagerly.
	if got := roots.len(); got != 1 {
		t.Errorf("got %d certs, want 1 (dir should be scanned when bundle has no roots)", got)
	}
}

func TestLoadOnDiskRootsScansDirWhenUserProvided(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a bundle with one cert.
	bundlePath := filepath.Join(tmpDir, "bundle.crt")
	bundleCA, _, err := generateCert("Bundle CA", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bundlePath,
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: bundleCA.Raw}), 0644); err != nil {
		t.Fatal(err)
	}

	// Create a directory with a different cert.
	certDir := filepath.Join(tmpDir, "certs")
	if err := os.MkdirAll(certDir, 0755); err != nil {
		t.Fatal(err)
	}
	dirCA, _, err := generateCert("Dir CA", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(certDir, "cert.pem"),
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: dirCA.Raw}), 0644); err != nil {
		t.Fatal(err)
	}

	// When user explicitly provides SSL_CERT_DIR, directories should be
	// scanned eagerly even though the bundle already has roots.
	roots, err := loadOnDiskRoots(bundlePath, certDir)
	if err != nil {
		t.Fatal(err)
	}

	// Should have 2 certs: one from bundle + one from user-provided dir.
	if got := roots.len(); got != 2 {
		t.Errorf("got %d certs, want 2 (user-provided dir should be scanned eagerly)", got)
	}
	// User-provided dirs are scanned eagerly — no lazy state should exist.
	if roots.lazyRoots != nil {
		t.Error("expected lazyRoots to be nil for user-provided dirs")
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

func TestLazyDirFallbackVerifiesCert(t *testing.T) {
	// CA only in directory (not bundle) — verification should trigger lazy
	// loading and succeed.
	bundleCA, _, err := generateCert("Bundle Dummy", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	dirCA, dirKey, _ := generateCert("Dir Only CA", true, nil, nil, nil)
	leaf, _, err := generateCert("leaf.example.com", false, dirCA, dirKey, nil)
	if err != nil {
		t.Fatal(err)
	}

	roots := setupLazyDirTest(t, bundleCA, dirCA)

	if roots.len() != 1 {
		t.Fatalf("expected 1 cert in pool, got %d", roots.len())
	}
	if roots.lazyRoots == nil {
		t.Fatal("expected lazyRoots to be set")
	}

	_, err = leaf.Verify(VerifyOptions{Roots: roots, CurrentTime: time.Now()})
	if err != nil {
		t.Fatalf("verification failed (lazy dir fallback did not work): %v", err)
	}

	// Confirm the lazy pool was created (check directly, not via load()).
	if roots.lazyRoots.pool == nil {
		t.Error("expected lazyPool to be set after verification")
	}
}
func TestLazyDirConcurrentAccess(t *testing.T) {
	bundleCA, bundleKey, err := generateCert("Bundle CA", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	dirCA, dirKey, err := generateCert("Lazy CA", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	bundleLeaf, _, err := generateCert("bundle.example.com", false, bundleCA, bundleKey, nil)
	if err != nil {
		t.Fatal(err)
	}
	lazyLeaf, _, err := generateCert("lazy.example.com", false, dirCA, dirKey, nil)
	if err != nil {
		t.Fatal(err)
	}

	roots := setupLazyDirTest(t, bundleCA, dirCA)

	// Concurrent verification: some hit bundle, some trigger lazy load.
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			if _, err := bundleLeaf.Verify(VerifyOptions{Roots: roots, CurrentTime: time.Now()}); err != nil {
				t.Errorf("bundle leaf verify failed: %v", err)
			}
		}()
		go func() {
			defer wg.Done()
			if _, err := lazyLeaf.Verify(VerifyOptions{Roots: roots, CurrentTime: time.Now()}); err != nil {
				t.Errorf("lazy leaf verify failed: %v", err)
			}
		}()
	}
	wg.Wait()
}

// setupLazyDirTest creates a bundle with bundleCA and a directory with dirCA,
// then calls loadOnDiskRoots with default paths. Returns the resulting pool.
func setupLazyDirTest(t *testing.T, bundleCA, dirCA *Certificate) *CertPool {
	t.Helper()
	tmpDir := t.TempDir()

	bundlePath := filepath.Join(tmpDir, "bundle.crt")
	if err := os.WriteFile(bundlePath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: bundleCA.Raw}), 0644); err != nil {
		t.Fatal(err)
	}

	certDir := filepath.Join(tmpDir, "certs")
	if err := os.MkdirAll(certDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(certDir, "dir-ca.pem"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: dirCA.Raw}), 0644); err != nil {
		t.Fatal(err)
	}

	origCertFiles, origCertDirectories := certFiles, certDirectories
	t.Cleanup(func() { certFiles = origCertFiles; certDirectories = origCertDirectories })
	certFiles = []string{bundlePath}
	certDirectories = []string{certDir}

	roots, err := loadOnDiskRoots("", "")
	if err != nil {
		t.Fatal(err)
	}
	return roots
}

func TestLazyDirWorksAfterClone(t *testing.T) {
	// CA only in directory, verified via Clone (simulates SystemCertPool()).
	bundleCA, _, err := generateCert("Bundle CA", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	dirCA, dirKey, _ := generateCert("Dir CA", true, nil, nil, nil)
	leaf, _, err := generateCert("clone.example.com", false, dirCA, dirKey, nil)
	if err != nil {
		t.Fatal(err)
	}

	roots := setupLazyDirTest(t, bundleCA, dirCA)
	cloned := roots.Clone()

	// Verify the clone preserved the lazy state (not eagerly resolved).
	if cloned.lazyRoots == nil {
		t.Fatal("clone lost lazyRoots")
	}
	if cloned.lazyRoots.pool != nil {
		t.Fatal("clone eagerly loaded lazy dirs")
	}
	if cloned.lazyRoots != roots.lazyRoots {
		t.Fatal("clone does not share lazyRoots pointer with original")
	}

	if _, err := leaf.Verify(VerifyOptions{Roots: cloned, CurrentTime: time.Now()}); err != nil {
		t.Fatalf("verification via cloned pool failed: %v", err)
	}
}

func TestLazyDirSameSubjectDifferentKey(t *testing.T) {
	// Bundle and directory have CAs with the same subject but different keys.
	// Leaf is signed by the directory CA's key.
	bundleCA, _, err := generateCert("Shared CA Name", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	dirCA, dirKey, _ := generateCert("Shared CA Name", true, nil, nil, nil)
	leaf, _, err := generateCert("samesubject.example.com", false, dirCA, dirKey, nil)
	if err != nil {
		t.Fatal(err)
	}

	roots := setupLazyDirTest(t, bundleCA, dirCA)

	if _, err := leaf.Verify(VerifyOptions{Roots: roots, CurrentTime: time.Now()}); err != nil {
		t.Fatalf("same-subject different-key verification failed: %v", err)
	}
}

func TestLazyDirDirectTrust(t *testing.T) {
	// A self-signed cert in the directory is directly trusted (matched via containsRoot).
	bundleCA, _, err := generateCert("Bundle CA", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	dirCert, _, err := generateCert("Directly Trusted", true, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	roots := setupLazyDirTest(t, bundleCA, dirCert)

	if _, err := dirCert.Verify(VerifyOptions{Roots: roots, CurrentTime: time.Now()}); err != nil {
		t.Fatalf("directly-trusted cert from lazy dir not recognized: %v", err)
	}
}
