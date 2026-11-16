// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/internal/upki"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"errors"
	"fmt"
	"internal/testenv"
	"math/big"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"

	"golang.org/x/crypto/cryptobyte"
)

const (
	tlsUPKITestLeafSerial       int64  = 42
	tlsUPKITestNoSCTSerial      int64  = 43
	tlsUPKITestMalformedSerial  int64  = 44
	tlsUPKITestOtherSerial      int64  = 45
	tlsUPKITestRootSerialOffset int64  = 1000
	tlsUPKITestSCTTime          uint64 = 1000
	tlsUPKITestCoverageLow      uint64 = 0
	tlsUPKITestCoverageHigh     uint64 = 2000
	tlsUPKITestServerName              = "test.golang.example"
	tlsUPKITestFilterName              = "test.filter"
)

var (
	tlsUPKITestLogID = [32]byte{1, 2, 3}
	tlsUPKITestTime  = time.Date(2026, time.November, 14, 12, 0, 0, 0, time.UTC)
)

// TestCheckUPKIRevocation exercises the default and overridden policy matrix,
// as well as malformed cache and certificate inputs, at the TLS policy boundary.
func TestCheckUPKIRevocation(t *testing.T) {
	t.Parallel()

	chain := newTLSUPKITestChain(t, tlsUPKITestLeafSerial, tlsUPKISCTExtensionValue(t, tlsUPKITestLogID, tlsUPKITestSCTTime))
	noSCTChain := newTLSUPKITestChain(t, tlsUPKITestNoSCTSerial, nil)
	malformedSCTChain := newTLSUPKITestChain(t, tlsUPKITestMalformedSerial, []byte{0x04, 0x05, 0})
	otherChain := newTLSUPKITestChain(t, tlsUPKITestOtherSerial, tlsUPKISCTExtensionValue(t, tlsUPKITestLogID, tlsUPKITestSCTTime))

	goodCache := writeTLSUPKITestCache(t, tlsUPKITestLogID, []tlsUPKITestBlock{{issuer: chain.issuerHash}})
	revokedCache := writeTLSUPKITestCache(t, tlsUPKITestLogID, []tlsUPKITestBlock{{issuer: chain.issuerHash, inverted: true}})
	notEnrolledCache := writeTLSUPKITestCache(t, tlsUPKITestLogID, []tlsUPKITestBlock{{issuer: otherChain.issuerHash}})
	notCoveredCache := writeTLSUPKITestCacheFiles(t, tlsUPKIEmptyIndex(), nil)
	malformedIndexCache := writeTLSUPKITestCacheFiles(t, []byte("not an index"), nil)
	malformedFilterCache := writeTLSUPKITestCacheFiles(t, tlsUPKIIndex(tlsUPKITestLogID), map[string][]byte{
		tlsUPKITestFilterName: []byte("not a filter"),
	})
	missingCache := filepath.Join(t.TempDir(), "missing")

	allowAll := UPKIRevocationConfig{
		Directory:   goodCache,
		MissingData: UPKIRevocationPolicyAllow,
		NotCovered:  UPKIRevocationPolicyAllow,
		MissingSCTs: UPKIRevocationPolicyAllow,
	}
	tests := []struct {
		name   string
		config *Config
		chains [][]*x509.Certificate
		// wantError, when set, must be matched by errors.Is. wantAnyError
		// requires an error without pinning which one, for cases whose error
		// is an unexported sentinel or a plain message.
		wantError    error
		wantAnyError bool
	}{
		{
			name:   "conclusively not revoked",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{Directory: goodCache}},
			chains: chain.verifiedChains(),
		},
		{
			name:      "revoked",
			config:    &Config{UPKIRevocation: UPKIRevocationConfig{Directory: revokedCache, NotCovered: UPKIRevocationPolicyAllow}},
			chains:    chain.verifiedChains(),
			wantError: ErrUPKIRevoked,
		},
		{
			name:   "disabled",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{Disabled: true, Directory: revokedCache}},
			chains: chain.verifiedChains(),
		},
		{
			name:   "no verified chains",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{Directory: revokedCache}},
		},
		{
			name:   "missing SCT default",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{Directory: missingCache}},
			chains: noSCTChain.verifiedChains(),
		},
		{
			name: "missing SCT allow",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory: missingCache, MissingSCTs: UPKIRevocationPolicyAllow,
			}},
			chains: noSCTChain.verifiedChains(),
		},
		{
			name: "missing SCT reject",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory: missingCache, MissingSCTs: UPKIRevocationPolicyReject,
			}},
			chains:       noSCTChain.verifiedChains(),
			wantAnyError: true,
		},
		{
			name:      "missing data default",
			config:    &Config{UPKIRevocation: UPKIRevocationConfig{Directory: missingCache}},
			chains:    chain.verifiedChains(),
			wantError: upki.ErrNoData,
		},
		{
			name: "missing data allow",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory: missingCache, MissingData: UPKIRevocationPolicyAllow,
			}},
			chains: chain.verifiedChains(),
		},
		{
			name:      "cache path is a file",
			config:    &Config{UPKIRevocation: UPKIRevocationConfig{Directory: filepath.Join(goodCache, upki.IndexFilename)}},
			chains:    chain.verifiedChains(),
			wantError: upki.ErrNoData,
		},
		{
			name: "cache path is a file allow",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory: filepath.Join(goodCache, upki.IndexFilename), MissingData: UPKIRevocationPolicyAllow,
			}},
			chains: chain.verifiedChains(),
		},
		{
			name:   "not covered default",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{Directory: notCoveredCache}},
			chains: chain.verifiedChains(),
		},
		{
			name: "not covered reject",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory: notCoveredCache, NotCovered: UPKIRevocationPolicyReject,
			}},
			chains:       chain.verifiedChains(),
			wantAnyError: true,
		},
		{
			name: "issuer not enrolled reject",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory: notEnrolledCache, NotCovered: UPKIRevocationPolicyReject,
			}},
			chains:       chain.verifiedChains(),
			wantAnyError: true,
		},
		{
			name:         "malformed SCT with explicit cache",
			config:       &Config{UPKIRevocation: allowAll},
			chains:       malformedSCTChain.verifiedChains(),
			wantAnyError: true,
		},
		{
			name: "malformed SCT with unavailable explicit cache",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory: missingCache, MissingData: UPKIRevocationPolicyAllow,
			}},
			chains:       malformedSCTChain.verifiedChains(),
			wantAnyError: true,
		},
		{
			name: "malformed index",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory:   malformedIndexCache,
				MissingData: UPKIRevocationPolicyAllow, NotCovered: UPKIRevocationPolicyAllow,
			}},
			chains:       chain.verifiedChains(),
			wantAnyError: true,
		},
		{
			name: "malformed filter",
			config: &Config{UPKIRevocation: UPKIRevocationConfig{
				Directory:   malformedFilterCache,
				MissingData: UPKIRevocationPolicyAllow, NotCovered: UPKIRevocationPolicyAllow,
			}},
			chains:       chain.verifiedChains(),
			wantAnyError: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := test.config.checkUPKIRevocation(test.chains)
			switch {
			case test.wantError != nil:
				if !errors.Is(err, test.wantError) {
					t.Fatalf("checkUPKIRevocation error = %v, want %v", err, test.wantError)
				}
			case test.wantAnyError:
				if err == nil {
					t.Fatal("checkUPKIRevocation succeeded, want an error")
				}
			case err != nil:
				t.Fatalf("checkUPKIRevocation: %v", err)
			}
		})
	}
}

// TestUPKIRevocationInvalidPolicy verifies that every policy field rejects
// values outside the public enumeration.
func TestUPKIRevocationInvalidPolicy(t *testing.T) {
	t.Parallel()

	chain := newTLSUPKITestChain(t, tlsUPKITestLeafSerial, nil)
	for _, field := range []string{"MissingData", "NotCovered", "MissingSCTs"} {
		t.Run(field, func(t *testing.T) {
			t.Parallel()
			config := UPKIRevocationConfig{}
			switch field {
			case "MissingData":
				config.MissingData = 255
			case "NotCovered":
				config.NotCovered = 255
			case "MissingSCTs":
				config.MissingSCTs = 255
			}
			err := (&Config{UPKIRevocation: config}).checkUPKIRevocation(chain.verifiedChains())
			if err == nil {
				t.Fatal("invalid policy was accepted")
			}
		})
	}
}

// TestUPKIPolicyReject covers the default-policy resolver independently of
// cache lookup, including its rejection of unknown policy values.
func TestUPKIPolicyReject(t *testing.T) {
	t.Parallel()

	tests := []struct {
		policy        UPKIRevocationPolicy
		defaultReject bool
		want          bool
		wantError     bool
	}{
		{UPKIRevocationPolicyDefault, false, false, false},
		{UPKIRevocationPolicyDefault, true, true, false},
		{UPKIRevocationPolicyAllow, true, false, false},
		{UPKIRevocationPolicyReject, false, true, false},
		{UPKIRevocationPolicy(255), false, false, true},
	}
	for _, test := range tests {
		name := fmt.Sprintf("policy=%d/default=%v", test.policy, test.defaultReject)
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			got, err := upkiPolicyReject(test.policy, test.defaultReject)
			if got != test.want || (err != nil) != test.wantError {
				t.Errorf("upkiPolicyReject = %v, %v; want %v, error=%v", got, err, test.want, test.wantError)
			}
		})
	}
}

// TestOpenUPKICacheIndex verifies ordered cache discovery and the no-cache
// result used by the missing-data policy.
func TestOpenUPKICacheIndex(t *testing.T) {
	t.Parallel()

	missing := filepath.Join(t.TempDir(), "missing")
	present := t.TempDir()
	if err := os.WriteFile(filepath.Join(present, upki.IndexFilename), []byte("index"), 0o644); err != nil {
		t.Fatal(err)
	}

	notDir := filepath.Join(present, upki.IndexFilename)
	dir, index, err := openUPKICacheIndex([]string{missing, notDir, present})
	if err != nil {
		t.Fatal(err)
	}
	if index == nil || dir != present {
		t.Fatalf("openUPKICacheIndex = %q, %v, want %q and a file", dir, index, present)
	}
	index.Close()

	dir, index, err = openUPKICacheIndex([]string{missing})
	if !errors.Is(err, upki.ErrNoData) || !errors.Is(err, os.ErrNotExist) || index != nil || dir != "" {
		t.Fatalf("openUPKICacheIndex(missing) = %q, %v, %v", dir, index, err)
	}

	t.Run("permission denied", func(t *testing.T) {
		blocked := t.TempDir()
		path := filepath.Join(blocked, upki.IndexFilename)
		if err := os.WriteFile(path, nil, 0); err != nil {
			t.Fatal(err)
		}
		if _, err := os.ReadFile(path); err == nil {
			t.Skip("process can read a mode-000 file")
		}
		_, _, err := openUPKICacheIndex([]string{blocked, missing})
		if !errors.Is(err, upki.ErrNoData) || !errors.Is(err, os.ErrPermission) {
			t.Fatalf("openUPKICacheIndex(blocked) = %v, want ErrNoData wrapping permission error", err)
		}
		dir, index, err := openUPKICacheIndex([]string{blocked, present})
		if err != nil {
			t.Fatal(err)
		}
		defer index.Close()
		if dir != present {
			t.Fatalf("selected %q, want %q", dir, present)
		}
	})
}

// TestUPKIRevocationDirectories records the platform-specific cache search
// order, including systems without a per-user cache directory.
func TestUPKIRevocationDirectories(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		goos        string
		userCache   string
		programData string
		want        []string
	}{
		{
			name:      "unix",
			goos:      "linux",
			userCache: filepath.Join("", "user-cache"),
			want: []string{
				filepath.Join("user-cache", "upki", "revocation"),
				"/var/cache/upki/revocation",
			},
		},
		{
			name:      "darwin",
			goos:      "darwin",
			userCache: filepath.Join("", "user-cache"),
			want: []string{
				filepath.Join("user-cache", "dev.rustls.upki", "revocation"),
				"/Library/Caches/dev.rustls.upki/revocation",
			},
		},
		{
			name:        "windows",
			goos:        "windows",
			userCache:   filepath.Join("", "user-cache"),
			programData: filepath.Join("", "program-data"),
			want: []string{
				filepath.Join("user-cache", "rustls", "upki", "cache", "revocation"),
				filepath.Join("program-data", "rustls", "upki", "cache", "revocation"),
			},
		},
		{
			name: "no user cache",
			goos: "linux",
			want: []string{"/var/cache/upki/revocation"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got := upkiRevocationDirectories(test.goos, test.userCache, test.programData)
			if !slices.Equal(got, test.want) {
				t.Errorf("upkiRevocationDirectories = %q, want %q", got, test.want)
			}
		})
	}
}

// TestUPKIRevocationDefaultDiscovery exercises platform cache discovery with
// a zero-valued UPKIRevocationConfig by pointing the user cache directory at
// a temporary cache. It is not parallel because it sets the environment.
func TestUPKIRevocationDefaultDiscovery(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "darwin", "ios", "plan9":
		t.Skipf("os.UserCacheDir does not honor XDG_CACHE_HOME on %s", runtime.GOOS)
	}

	chain := newTLSUPKITestChain(t, tlsUPKITestLeafSerial, tlsUPKISCTExtensionValue(t, tlsUPKITestLogID, tlsUPKITestSCTTime))
	setTLSUPKIUserCache(t, chain)

	clientConfig := chain.clientConfig("")
	if err := clientConfig.checkUPKIRevocation(chain.verifiedChains()); !errors.Is(err, ErrUPKIRevoked) {
		t.Fatalf("checkUPKIRevocation = %v, want ErrUPKIRevoked", err)
	}
	clientErr, _ := runTLSUPKIHandshake(t, clientConfig, chain.serverConfig())
	if !errors.Is(clientErr, ErrUPKIRevoked) {
		t.Fatalf("handshake error = %v, want ErrUPKIRevoked", clientErr)
	}

	malformed := newTLSUPKITestChain(t, tlsUPKITestMalformedSerial, []byte{0x04, 0x05, 0})
	clientConfig = malformed.clientConfig("")
	clientErr, _ = runTLSUPKIHandshake(t, clientConfig, malformed.serverConfig())
	var verificationErr *CertificateVerificationError
	if !errors.As(clientErr, &verificationErr) {
		t.Fatalf("malformed SCT with a discovered cache: got %v, want certificate verification error", clientErr)
	}

	// Without a user cache, discovery falls through to the system cache,
	// which may exist on the machine running the tests.
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	if _, err := os.Stat("/var/cache/upki/revocation/index.bin"); err == nil {
		t.Log("system cache present; skipping the no-cache assertion")
		return
	}
	clientConfig = chain.clientConfig("")
	if err := clientConfig.checkUPKIRevocation(chain.verifiedChains()); err != nil {
		t.Fatalf("checkUPKIRevocation with no cache = %v, want nil", err)
	}
	if clientErr, serverErr := runTLSUPKIHandshake(t, malformed.clientConfig(""), malformed.serverConfig()); clientErr != nil || serverErr != nil {
		t.Fatalf("malformed SCT without a cache: client=%v, server=%v", clientErr, serverErr)
	}
	clientConfig.UPKIRevocation = UPKIRevocationConfig{MissingData: UPKIRevocationPolicyReject}
	if err := clientConfig.checkUPKIRevocation(chain.verifiedChains()); !errors.Is(err, upki.ErrNoData) {
		t.Fatalf("required cache = %v, want ErrNoData", err)
	}
}

// setTLSUPKIUserCache points the user cache directory at a temporary cache
// in which chain's certificate is revoked. It sets the environment, so the
// calling test must not be parallel.
func setTLSUPKIUserCache(t *testing.T, chain tlsUPKITestChain) {
	t.Helper()

	userCache := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", userCache)
	cacheDir := filepath.Join(userCache, "upki", "revocation")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeTLSUPKITestCacheFilesIn(t, cacheDir, tlsUPKIIndex(tlsUPKITestLogID), map[string][]byte{
		tlsUPKITestFilterName: tlsUPKIFilter(tlsUPKITestLogID, []tlsUPKITestBlock{{issuer: chain.issuerHash, inverted: true}}),
	})
}

// TestUPKIRevocationGODEBUG verifies that tlsupki=0 disables checking for
// configurations that rely on platform-default discovery and leaves an
// explicitly configured directory alone. It is not parallel because it sets
// the environment.
func TestUPKIRevocationGODEBUG(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "darwin", "ios", "plan9":
		t.Skipf("os.UserCacheDir does not honor XDG_CACHE_HOME on %s", runtime.GOOS)
	}

	chain := newTLSUPKITestChain(t, tlsUPKITestLeafSerial, tlsUPKISCTExtensionValue(t, tlsUPKITestLogID, tlsUPKITestSCTTime))
	setTLSUPKIUserCache(t, chain)
	explicitCache := writeTLSUPKITestCache(t, tlsUPKITestLogID, []tlsUPKITestBlock{{issuer: chain.issuerHash, inverted: true}})

	tests := []struct {
		name      string
		godebug   string
		config    UPKIRevocationConfig
		wantError error
	}{
		{name: "default discovery", godebug: "tlsupki=1", wantError: ErrUPKIRevoked},
		{name: "default discovery disabled", godebug: "tlsupki=0"},
		{name: "explicit directory", godebug: "tlsupki=1", config: UPKIRevocationConfig{Directory: explicitCache}, wantError: ErrUPKIRevoked},
		{name: "explicit directory not disabled", godebug: "tlsupki=0", config: UPKIRevocationConfig{Directory: explicitCache}, wantError: ErrUPKIRevoked},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testenv.SetGODEBUG(t, test.godebug)
			clientConfig := chain.clientConfig("")
			clientConfig.UPKIRevocation = test.config
			err := clientConfig.checkUPKIRevocation(chain.verifiedChains())
			if test.wantError != nil && !errors.Is(err, test.wantError) {
				t.Fatalf("checkUPKIRevocation = %v, want %v", err, test.wantError)
			}
			if test.wantError == nil && err != nil {
				t.Fatalf("checkUPKIRevocation = %v, want nil", err)
			}
			clientErr, _ := runTLSUPKIHandshake(t, clientConfig, chain.serverConfig())
			if test.wantError != nil && !errors.Is(clientErr, test.wantError) {
				t.Fatalf("handshake error = %v, want %v", clientErr, test.wantError)
			}
			if test.wantError == nil && clientErr != nil {
				t.Fatalf("handshake error = %v, want nil", clientErr)
			}
		})
	}
}

// TestUPKIRevocationHandshake verifies the ordering and error wrapping of
// revocation checks in the client handshake.
func TestUPKIRevocationHandshake(t *testing.T) {
	t.Parallel()

	chain := newTLSUPKITestChain(t, tlsUPKITestLeafSerial, tlsUPKISCTExtensionValue(t, tlsUPKITestLogID, tlsUPKITestSCTTime))
	revokedCache := writeTLSUPKITestCache(t, tlsUPKITestLogID, []tlsUPKITestBlock{{issuer: chain.issuerHash, inverted: true}})
	goodCache := writeTLSUPKITestCache(t, tlsUPKITestLogID, []tlsUPKITestBlock{{issuer: chain.issuerHash}})
	malformedCache := writeTLSUPKITestCacheFiles(t, []byte("not an index"), nil)

	t.Run("revoked before callbacks", func(t *testing.T) {
		t.Parallel()
		var callbacks []string
		clientConfig := chain.clientConfig(revokedCache)
		clientConfig.VerifyPeerCertificate = func([][]byte, [][]*x509.Certificate) error {
			callbacks = append(callbacks, "VerifyPeerCertificate")

			return nil
		}
		clientConfig.VerifyConnection = func(ConnectionState) error {
			callbacks = append(callbacks, "VerifyConnection")

			return nil
		}
		clientErr, _ := runTLSUPKIHandshake(t, clientConfig, chain.serverConfig())
		if !errors.Is(clientErr, ErrUPKIRevoked) {
			t.Fatalf("client error = %v, want ErrUPKIRevoked", clientErr)
		}
		var verificationError *CertificateVerificationError
		if !errors.As(clientErr, &verificationError) {
			t.Fatalf("client error %T, want *CertificateVerificationError", clientErr)
		}
		if len(callbacks) != 0 {
			t.Errorf("callbacks ran before revocation rejection: %v", callbacks)
		}
	})

	t.Run("accepted before callbacks", func(t *testing.T) {
		t.Parallel()
		var callbacks []string
		clientConfig := chain.clientConfig(goodCache)
		clientConfig.VerifyPeerCertificate = func([][]byte, [][]*x509.Certificate) error {
			callbacks = append(callbacks, "VerifyPeerCertificate")

			return nil
		}
		clientConfig.VerifyConnection = func(ConnectionState) error {
			callbacks = append(callbacks, "VerifyConnection")

			return nil
		}
		clientErr, serverErr := runTLSUPKIHandshake(t, clientConfig, chain.serverConfig())
		if clientErr != nil || serverErr != nil {
			t.Fatalf("handshake errors: client=%v server=%v", clientErr, serverErr)
		}
		if want := []string{"VerifyPeerCertificate", "VerifyConnection"}; !slices.Equal(callbacks, want) {
			t.Errorf("callbacks = %v, want %v", callbacks, want)
		}
	})

	t.Run("normal verification happens first", func(t *testing.T) {
		t.Parallel()
		clientConfig := chain.clientConfig(malformedCache)
		clientConfig.ServerName = "wrong.golang.example"
		clientErr, _ := runTLSUPKIHandshake(t, clientConfig, chain.serverConfig())
		var verificationError *CertificateVerificationError
		if !errors.As(clientErr, &verificationError) {
			t.Fatalf("client error %v, want *CertificateVerificationError", clientErr)
		}
		var hostnameError x509.HostnameError
		if !errors.As(verificationError.Err, &hostnameError) {
			t.Fatalf("verification error = %v, want x509.HostnameError", verificationError.Err)
		}
	})

	t.Run("InsecureSkipVerify bypasses revocation", func(t *testing.T) {
		t.Parallel()
		clientConfig := chain.clientConfig(revokedCache)
		clientConfig.InsecureSkipVerify = true
		clientErr, serverErr := runTLSUPKIHandshake(t, clientConfig, chain.serverConfig())
		if clientErr != nil || serverErr != nil {
			t.Fatalf("handshake errors: client=%v server=%v", clientErr, serverErr)
		}
	})
}

// TestUPKIRevocationDoesNotCheckClientCertificates ensures that the client-only
// feature does not accidentally apply server-side policy to client certificates.
func TestUPKIRevocationDoesNotCheckClientCertificates(t *testing.T) {
	t.Parallel()

	clientConfig := testConfigClient()
	serverConfig := testConfigServer()
	serverConfig.ClientAuth = RequireAndVerifyClientCert
	serverConfig.ClientCAs = testClientRootCertPool
	serverConfig.UPKIRevocation = UPKIRevocationConfig{MissingSCTs: UPKIRevocationPolicyReject}
	if _, _, err := testHandshake(t, clientConfig, serverConfig); err != nil {
		t.Fatalf("handshake: %v", err)
	}
}

// TestUPKIRevocationResumption verifies that a newly revoked cached certificate
// prevents resumption and causes a full handshake with the replacement chain.
func TestUPKIRevocationResumption(t *testing.T) {
	for _, version := range []uint16{VersionTLS12, VersionTLS13} {
		t.Run(VersionName(version), func(t *testing.T) {
			t.Parallel()

			oldChain := newTLSUPKITestChain(t, tlsUPKITestLeafSerial, tlsUPKISCTExtensionValue(t, tlsUPKITestLogID, tlsUPKITestSCTTime))
			newChain := newTLSUPKITestChain(t, tlsUPKITestNoSCTSerial, tlsUPKISCTExtensionValue(t, tlsUPKITestLogID, tlsUPKITestSCTTime))
			cache := writeTLSUPKITestCache(t, tlsUPKITestLogID, []tlsUPKITestBlock{
				{issuer: oldChain.issuerHash},
				{issuer: newChain.issuerHash},
			})

			roots := x509.NewCertPool()
			roots.AddCert(oldChain.root)
			roots.AddCert(newChain.root)
			clientConfig := oldChain.clientConfig(cache)
			clientConfig.RootCAs = roots
			sessionCache := &countingSessionCache{ClientSessionCache: NewLRUClientSessionCache(1)}
			clientConfig.ClientSessionCache = sessionCache
			clientConfig.MinVersion, clientConfig.MaxVersion = version, version
			serverConfig := oldChain.serverConfig()
			serverConfig.MinVersion, serverConfig.MaxVersion = version, version

			_, state, err := testHandshake(t, clientConfig, serverConfig)
			if err != nil || state.DidResume {
				t.Fatalf("initial handshake = resume %v, error %v", state.DidResume, err)
			}
			_, state, err = testHandshake(t, clientConfig, serverConfig)
			if err != nil || !state.DidResume {
				t.Fatalf("control resumption = resume %v, error %v", state.DidResume, err)
			}

			filter := tlsUPKIFilter(tlsUPKITestLogID, []tlsUPKITestBlock{
				{issuer: oldChain.issuerHash, inverted: true},
				{issuer: newChain.issuerHash},
			})
			if err := os.WriteFile(filepath.Join(cache, tlsUPKITestFilterName), filter, 0o644); err != nil {
				t.Fatal(err)
			}
			serverConfig.Certificates = []Certificate{newChain.certificate}

			_, state, err = testHandshake(t, clientConfig, serverConfig)
			if err != nil {
				t.Fatalf("replacement handshake: %v", err)
			}
			if state.DidResume {
				t.Error("resumed a session whose cached certificate is now revoked")
			}
			if got := state.PeerCertificates[0].SerialNumber.Int64(); got != tlsUPKITestNoSCTSerial {
				t.Errorf("replacement serial = %d, want %d", got, tlsUPKITestNoSCTSerial)
			}
			if sessionCache.deletions != 1 {
				t.Errorf("session cache deletions = %d, want 1 for the revoked session", sessionCache.deletions)
			}
		})
	}
}

// countingSessionCache counts the deletions (a Put with a nil session) the
// client performs on the wrapped cache.
type countingSessionCache struct {
	ClientSessionCache
	deletions int
}

func (c *countingSessionCache) Put(key string, session *ClientSessionState) {
	if session == nil {
		c.deletions++
	}
	c.ClientSessionCache.Put(key, session)
}

// runTLSUPKIHandshake performs both halves of an in-memory handshake and
// returns each peer's error so tests can inspect the client failure precisely.
func runTLSUPKIHandshake(t *testing.T, clientConfig, serverConfig *Config) (clientErr, serverErr error) {
	t.Helper()

	clientConn, serverConn := localPipe(t)
	clientDone := make(chan error, 1)
	go func() {
		clientDone <- Client(clientConn, clientConfig).Handshake()
		clientConn.Close()
	}()
	serverErr = Server(serverConn, serverConfig).Handshake()
	serverConn.Close()
	clientErr = <-clientDone

	return clientErr, serverErr
}

type tlsUPKITestChain struct {
	certificate Certificate
	leaf        *x509.Certificate
	root        *x509.Certificate
	issuerHash  [32]byte
}

// newTLSUPKITestChain creates a root and server certificate whose verified
// chain, TLS configurations, and issuer hash are all available to a test.
func newTLSUPKITestChain(t *testing.T, serial int64, sctExtension []byte) tlsUPKITestChain {
	t.Helper()

	rootKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	rootTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(serial + tlsUPKITestRootSerialOffset),
		Subject:               pkix.Name{CommonName: "UPKI test root"},
		NotBefore:             tlsUPKITestTime.Add(-time.Hour),
		NotAfter:              tlsUPKITestTime.Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	rootDER, err := x509.CreateCertificate(rand.Reader, rootTemplate, rootTemplate, &rootKey.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	root, err := x509.ParseCertificate(rootDER)
	if err != nil {
		t.Fatal(err)
	}

	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	leafTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(serial),
		Subject:      pkix.Name{CommonName: tlsUPKITestServerName},
		DNSNames:     []string{tlsUPKITestServerName},
		NotBefore:    tlsUPKITestTime.Add(-time.Hour),
		NotAfter:     tlsUPKITestTime.Add(time.Hour),
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	if sctExtension != nil {
		leafTemplate.ExtraExtensions = []pkix.Extension{{
			Id:    []int{1, 3, 6, 1, 4, 1, 11129, 2, 4, 2},
			Value: sctExtension,
		}}
	}
	leafDER, err := x509.CreateCertificate(rand.Reader, leafTemplate, root, &leafKey.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	leaf, err := x509.ParseCertificate(leafDER)
	if err != nil {
		t.Fatal(err)
	}

	return tlsUPKITestChain{
		certificate: Certificate{Certificate: [][]byte{leafDER}, PrivateKey: leafKey, Leaf: leaf},
		leaf:        leaf,
		root:        root,
		issuerHash:  sha256.Sum256(root.RawSubjectPublicKeyInfo),
	}
}

func (chain tlsUPKITestChain) verifiedChains() [][]*x509.Certificate {
	return [][]*x509.Certificate{{chain.leaf, chain.root}}
}

func (chain tlsUPKITestChain) clientConfig(cache string) *Config {
	roots := x509.NewCertPool()
	roots.AddCert(chain.root)

	return &Config{
		RootCAs:    roots,
		ServerName: tlsUPKITestServerName,
		Time:       func() time.Time { return tlsUPKITestTime },
		UPKIRevocation: UPKIRevocationConfig{
			Directory: cache,
		},
	}
}

func (chain tlsUPKITestChain) serverConfig() *Config {
	return &Config{
		Certificates: []Certificate{chain.certificate},
		Time:         func() time.Time { return tlsUPKITestTime },
	}
}

// writeTLSUPKITestCache writes the minimal matching index and filter pair used
// by most TLS tests.
func writeTLSUPKITestCache(t *testing.T, logID [32]byte, blocks []tlsUPKITestBlock) string {
	t.Helper()

	return writeTLSUPKITestCacheFiles(t, tlsUPKIIndex(logID), map[string][]byte{
		tlsUPKITestFilterName: tlsUPKIFilter(logID, blocks),
	})
}

// writeTLSUPKITestCacheFiles materializes an arbitrary cache layout so tests
// can also exercise malformed and missing files.
func writeTLSUPKITestCacheFiles(t *testing.T, index []byte, filters map[string][]byte) string {
	t.Helper()
	dir := t.TempDir()
	writeTLSUPKITestCacheFilesIn(t, dir, index, filters)

	return dir
}

// writeTLSUPKITestCacheFilesIn writes a cache layout into an existing dir.
func writeTLSUPKITestCacheFilesIn(t *testing.T, dir string, index []byte, filters map[string][]byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, upki.IndexFilename), index, 0o644); err != nil {
		t.Fatal(err)
	}
	for name, data := range filters {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

type tlsUPKITestBlock struct {
	issuer   [32]byte
	inverted bool
}

// tlsUPKIFilter encodes the narrow rank-zero Clubcard form needed by these
// tests. It follows C2SP upki-revocation v1.0.0:
// https://c2sp.org/upki-revocation@v1.0.0#revocation-filter-files.
//
// The TLS tests cannot reuse crypto/internal/upki's _test.go encoder without
// turning test-only code into a shipped package, so this copy intentionally
// supports only one coverage interval and rank-zero blocks.
func tlsUPKIFilter(logID [32]byte, blocks []tlsUPKITestBlock) []byte {
	var builder cryptobyte.Builder
	builder.AddUint8(4)
	builder.AddUint8(0)
	builder.AddUint16(1)
	builder.AddBytes(logID[:])
	builder.AddUint64(tlsUPKITestCoverageLow)
	builder.AddUint64(tlsUPKITestCoverageHigh)
	builder.AddUint32(uint32(len(blocks)))
	for _, block := range blocks {
		builder.AddBytes(block.issuer[:])
		builder.AddUint32(0)
		builder.AddUint8(0)
		builder.AddUint32(0)
		builder.AddUint32(0)
		builder.AddUint32(0)
		if block.inverted {
			builder.AddUint8(1)
		} else {
			builder.AddUint8(0)
		}
		builder.AddUint16(0)
	}
	builder.AddUint8(0)
	builder.AddUint32(0)

	return builder.BytesOrPanic()
}

// tlsUPKIIndex encodes a one-filter index as specified by C2SP
// upki-revocation v1.0.0:
// https://c2sp.org/upki-revocation@v1.0.0#the-index.
func tlsUPKIIndex(logID [32]byte) []byte {
	var builder cryptobyte.Builder
	builder.AddBytes([]byte("upkiidx1"))
	builder.AddUint16(1)
	builder.AddUint32(1)
	filename := make([]byte, 32)
	copy(filename, tlsUPKITestFilterName)
	builder.AddBytes(filename)
	builder.AddBytes(logID[:])
	builder.AddUint64(14 + 32 + 42)
	builder.AddUint16(1)
	builder.AddUint16(0)
	builder.AddUint64(tlsUPKITestCoverageLow)
	builder.AddUint64(tlsUPKITestCoverageHigh)

	return builder.BytesOrPanic()
}

// tlsUPKIEmptyIndex encodes a valid index with no filter files.
func tlsUPKIEmptyIndex() []byte {
	var builder cryptobyte.Builder
	builder.AddBytes([]byte("upkiidx1"))
	builder.AddUint16(0)
	builder.AddUint32(0)

	return builder.BytesOrPanic()
}

// tlsUPKISCTExtensionValue encodes an RFC 6962 §3.3
// SignedCertificateTimestampList containing one v1 SCT. The signature is
// omitted because revocation lookup uses only the log ID and timestamp.
func tlsUPKISCTExtensionValue(t *testing.T, logID [32]byte, timestamp uint64) []byte {
	t.Helper()

	var serialized cryptobyte.Builder
	serialized.AddUint8(0)
	serialized.AddBytes(logID[:])
	serialized.AddUint64(timestamp)
	serialized.AddUint16(0)
	serializedBytes := serialized.BytesOrPanic()

	var contents cryptobyte.Builder
	contents.AddUint16(uint16(2 + len(serializedBytes)))
	contents.AddUint16(uint16(len(serializedBytes)))
	contents.AddBytes(serializedBytes)

	var extension cryptobyte.Builder
	extension.AddASN1OctetString(contents.BytesOrPanic())
	encoded, err := extension.Bytes()
	if err != nil {
		t.Fatal(err)
	}

	return encoded
}
