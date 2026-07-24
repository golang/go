// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package web

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"cmd/go/internal/auth"
)

func TestUserAgent(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(r.UserAgent()))
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal("parse httptest url:", err)
	}
	res, err := Get(Insecure, u)
	if err != nil {
		t.Error("http get:", err)
	}
	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Error("read response body:", err)
	}
	gotUserAgent := string(bytes.TrimSpace(b))
	if gotUserAgent != userAgent {
		t.Errorf("User-Agent: %s, want %s", gotUserAgent, userAgent)
	}
}

func TestMTLSClient(t *testing.T) {
	for _, test := range []struct {
		name     string
		combined bool
	}{
		{name: "separate certificate and key files", combined: false},
		{name: "combined certificate and key file", combined: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			receivedCertificate := make(chan bool, 1)
			ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedCertificate <- len(r.TLS.PeerCertificates) > 0
			}))
			clientCert, clientCAs := newClientCertificate(t)
			ts.TLS = &tls.Config{ClientAuth: tls.RequireAndVerifyClientCert, ClientCAs: clientCAs}
			ts.StartTLS()
			defer ts.Close()

			var certFile, keyFile string
			if test.combined {
				certFile = writeCombinedCertificateFile(t, clientCert)
				keyFile = certFile
			} else {
				certFile, keyFile = writeCertificateFiles(t, clientCert)
			}
			u, err := url.Parse(ts.URL)
			if err != nil {
				t.Fatal(err)
			}
			cert := auth.ClientCertificate{Origin: u.Scheme + "://" + u.Host, CertFile: certFile, KeyFile: keyFile}
			client := mtlsHTTPClientWithLookup(ts.Client(), false, lookupClientCertificate(cert))
			res, err := client.Get(ts.URL)
			if err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
			if !<-receivedCertificate {
				t.Error("server did not receive a client certificate")
			}
		})
	}
}

func TestMTLSClientLazyForOtherHost(t *testing.T) {
	ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ts.Close()

	cert := auth.ClientCertificate{
		Origin:   "https://mtls.example.com:443",
		CertFile: "missing-client-cert.pem",
		KeyFile:  "missing-client-key.pem",
	}
	client := mtlsHTTPClientWithLookup(ts.Client(), false, lookupClientCertificate(cert))
	res, err := client.Get(ts.URL)
	if err != nil {
		t.Fatalf("request to an unrelated host unexpectedly loaded the certificate: %v", err)
	}
	res.Body.Close()
}

func TestMTLSClientMissingCertificate(t *testing.T) {
	ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	cert := auth.ClientCertificate{
		Origin:   u.Scheme + "://" + u.Host,
		CertFile: "missing-client-cert.pem",
		KeyFile:  "missing-client-key.pem",
	}
	client := mtlsHTTPClientWithLookup(ts.Client(), false, lookupClientCertificate(cert))
	_, err = client.Get(ts.URL)
	if err == nil || !strings.Contains(err.Error(), "loading GOAUTH=mtls client certificate") {
		t.Fatalf("client.Get error = %v, want missing GOAUTH=mtls certificate error", err)
	}
}

func TestMTLSClientRedirectDoesNotForwardCertificate(t *testing.T) {
	destinationReceivedCertificate := make(chan bool, 1)
	destination := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		destinationReceivedCertificate <- len(r.TLS.PeerCertificates) > 0
	}))
	destination.TLS = &tls.Config{ClientAuth: tls.RequestClientCert}
	destination.StartTLS()
	defer destination.Close()

	destinationURL, err := url.Parse(destination.URL)
	if err != nil {
		t.Fatal(err)
	}
	destinationURL.Host = "other.example.com:" + destinationURL.Port()

	source := httptest.NewUnstartedServer(http.RedirectHandler(destinationURL.String(), http.StatusFound))
	source.TLS = &tls.Config{ClientAuth: tls.RequireAnyClientCert}
	source.StartTLS()
	defer source.Close()

	certFile, keyFile := writeCertificateFiles(t, source.TLS.Certificates[0])
	sourceURL, err := url.Parse(source.URL)
	if err != nil {
		t.Fatal(err)
	}
	sourceURL.Host = "mtls.example.com:" + sourceURL.Port()
	baseClient := source.Client()
	transport := baseClient.Transport.(*http.Transport).Clone()
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		_, port, err := net.SplitHostPort(address)
		if err != nil {
			return nil, err
		}
		return (&net.Dialer{}).DialContext(ctx, network, net.JoinHostPort("127.0.0.1", port))
	}
	baseClient.Transport = transport
	cert := auth.ClientCertificate{Origin: sourceURL.Scheme + "://" + sourceURL.Host, CertFile: certFile, KeyFile: keyFile}
	client := mtlsHTTPClientWithLookup(baseClient, false, lookupClientCertificate(cert))
	res, err := client.Get(sourceURL.String())
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if <-destinationReceivedCertificate {
		t.Error("redirect destination received the client certificate")
	}
}

func TestMTLSClientRejectsHTTPSProxy(t *testing.T) {
	ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer ts.Close()

	certFile, keyFile := writeCertificateFiles(t, ts.TLS.Certificates[0])
	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	baseClient := ts.Client()
	transport := baseClient.Transport.(*http.Transport).Clone()
	transport.Proxy = func(*http.Request) (*url.URL, error) {
		return url.Parse("https://proxy.example.com")
	}
	baseClient.Transport = transport
	cert := auth.ClientCertificate{Origin: u.Scheme + "://" + u.Host, CertFile: certFile, KeyFile: keyFile}
	client := mtlsHTTPClientWithLookup(baseClient, false, lookupClientCertificate(cert))
	_, err = client.Get(ts.URL)
	if err == nil || !strings.Contains(err.Error(), "GOAUTH=mtls does not support HTTPS proxy") {
		t.Fatalf("client.Get error = %v, want HTTPS proxy error", err)
	}
}

func TestMTLSClientRejectsGOINSECURE(t *testing.T) {
	cert := auth.ClientCertificate{
		Origin:   "https://mtls.example.com:443",
		CertFile: "missing-client-cert.pem",
		KeyFile:  "missing-client-key.pem",
	}
	client := mtlsHTTPClientWithLookup(http.DefaultClient, true, lookupClientCertificate(cert))
	_, err := client.Get("https://mtls.example.com")
	if err == nil || !strings.Contains(err.Error(), "GOAUTH=mtls cannot be used with GOINSECURE") {
		t.Fatalf("client.Get error = %v, want GOINSECURE error", err)
	}
}

func lookupClientCertificate(cert auth.ClientCertificate) clientCertificateLookup {
	return func(req *http.Request) (auth.ClientCertificate, bool) {
		host := req.URL.Host
		if req.Host != "" {
			host = req.Host
		}
		u := &url.URL{Scheme: req.URL.Scheme, Host: host}
		port := u.Port()
		if port == "" {
			port = "443"
		}
		origin := u.Scheme + "://" + net.JoinHostPort(strings.ToLower(strings.TrimSuffix(u.Hostname(), ".")), port)
		return cert, origin == cert.Origin
	}
}

func newClientCertificate(t *testing.T) (tls.Certificate, *x509.CertPool) {
	t.Helper()

	now := time.Now()
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	caTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		NotBefore:             now.Add(-time.Hour),
		NotAfter:              now.Add(time.Hour),
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	caDER, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	ca, err := x509.ParseCertificate(caDER)
	if err != nil {
		t.Fatal(err)
	}

	clientKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	clientTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		NotBefore:    now.Add(-time.Hour),
		NotAfter:     now.Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	clientDER, err := x509.CreateCertificate(rand.Reader, clientTemplate, ca, &clientKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}

	pool := x509.NewCertPool()
	pool.AddCert(ca)
	return tls.Certificate{Certificate: [][]byte{clientDER, caDER}, PrivateKey: clientKey}, pool
}

func writeCertificateFiles(t *testing.T, cert tls.Certificate) (certFile, keyFile string) {
	t.Helper()

	var certPEM bytes.Buffer
	for _, der := range cert.Certificate {
		if err := pem.Encode(&certPEM, &pem.Block{Type: "CERTIFICATE", Bytes: der}); err != nil {
			t.Fatal(err)
		}
	}
	key, err := x509.MarshalPKCS8PrivateKey(cert.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key})

	dir := t.TempDir()
	certFile = dir + "/client-cert.pem"
	keyFile = dir + "/client-key.pem"
	if err := os.WriteFile(certFile, certPEM.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keyFile, keyPEM, 0600); err != nil {
		t.Fatal(err)
	}
	return certFile, keyFile
}

func writeCombinedCertificateFile(t *testing.T, cert tls.Certificate) string {
	t.Helper()

	var combined bytes.Buffer
	for _, der := range cert.Certificate {
		if err := pem.Encode(&combined, &pem.Block{Type: "CERTIFICATE", Bytes: der}); err != nil {
			t.Fatal(err)
		}
	}
	key, err := x509.MarshalPKCS8PrivateKey(cert.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	if err := pem.Encode(&combined, &pem.Block{Type: "PRIVATE KEY", Bytes: key}); err != nil {
		t.Fatal(err)
	}

	file := t.TempDir() + "/client.pem"
	if err := os.WriteFile(file, combined.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return file
}
