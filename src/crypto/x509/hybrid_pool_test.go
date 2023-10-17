// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509_test

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"internal/testenv"
	"math/big"
	"runtime"
	"testing"
	"time"
)

func TestHybridPool(t *testing.T) {
	t.Parallel()
	if !(runtime.GOOS == "windows" || runtime.GOOS == "darwin" || runtime.GOOS == "ios") {
		t.Skipf("platform verifier not available on %s", runtime.GOOS)
	}
	if !testenv.HasExternalNetwork() {
		t.Skip()
	}
	if runtime.GOOS == "windows" {
		// NOTE(#51599): on the Windows builders we sometimes see that the state
		// of the root pool is not fully initialized, causing an expected
		// platform verification to fail. In part this is because Windows
		// dynamically populates roots into its local trust store at time of
		// use. We can attempt to prime the pool by attempting TLS connections
		// to google.com until it works, suggesting the pool has been properly
		// updated. If after we hit the dealine, the pool has _still_ not been
		// populated with the expected root, it's unlikely we are ever going to
		// get into a good state, and so we just fail the test. #52108 suggests
		// a better possible long term solution.

		deadline := time.Now().Add(time.Second * 10)
		nextSleep := 10 * time.Millisecond
		for i := 0; ; i++ {
			c, err := tls.Dial("tcp", "google.com:443", nil)
			if err == nil {
				c.Close()
				break
			}
			nextSleep = nextSleep * time.Duration(i)
			if time.Until(deadline) < nextSleep {
				t.Fatal("windows root pool appears to be in an uninitialized state (missing root that chains to google.com)")
			}
			time.Sleep(nextSleep)
		}
	}

	// Get the google.com chain, which should be valid on all platforms we
	// are testing
	c, err := tls.Dial("tcp", "google.com:443", &tls.Config{InsecureSkipVerify: true})
	if err != nil {
		t.Fatalf("tls connection failed: %s", err)
	}
	googChain := c.ConnectionState().PeerCertificates

	rootTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "Go test root"},
		IsCA:                  true,
		BasicConstraintsValid: true,
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour * 10),
	}
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("failed to generate test key: %s", err)
	}
	rootDER, err := x509.CreateCertificate(rand.Reader, rootTmpl, rootTmpl, k.Public(), k)
	if err != nil {
		t.Fatalf("failed to create test cert: %s", err)
	}
	root, err := x509.ParseCertificate(rootDER)
	if err != nil {
		t.Fatalf("failed to parse test cert: %s", err)
	}

	pool, err := x509.SystemCertPool()
	if err != nil {
		t.Fatalf("SystemCertPool failed: %s", err)
	}
	opts := x509.VerifyOptions{Roots: pool}

	_, err = googChain[0].Verify(opts)
	if err != nil {
		t.Fatalf("verification failed for google.com chain (system only pool): %s", err)
	}

	pool.AddCert(root)

	_, err = googChain[0].Verify(opts)
	if err != nil {
		t.Fatalf("verification failed for google.com chain (hybrid pool): %s", err)
	}

	certTmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour * 10),
		DNSNames:     []string{"example.com"},
	}
	certDER, err := x509.CreateCertificate(rand.Reader, certTmpl, rootTmpl, k.Public(), k)
	if err != nil {
		t.Fatalf("failed to create test cert: %s", err)
	}
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		t.Fatalf("failed to parse test cert: %s", err)
	}

	_, err = cert.Verify(opts)
	if err != nil {
		t.Fatalf("verification failed for custom chain (hybrid pool): %s", err)
	}
}
