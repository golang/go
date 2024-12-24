// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/internal/cryptotest"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"internal/byteorder"
	"internal/testenv"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/crypto/cryptobyte"
)

var (
	port   = flag.String("port", "", "")
	server = flag.Bool("server", false, "")

	isHandshakerSupported = flag.Bool("is-handshaker-supported", false, "")

	keyfile  = flag.String("key-file", "", "")
	certfile = flag.String("cert-file", "", "")

	trustCert = flag.String("trust-cert", "", "")

	minVersion    = flag.Int("min-version", VersionSSL30, "")
	maxVersion    = flag.Int("max-version", VersionTLS13, "")
	expectVersion = flag.Int("expect-version", 0, "")

	noTLS1  = flag.Bool("no-tls1", false, "")
	noTLS11 = flag.Bool("no-tls11", false, "")
	noTLS12 = flag.Bool("no-tls12", false, "")
	noTLS13 = flag.Bool("no-tls13", false, "")

	requireAnyClientCertificate = flag.Bool("require-any-client-certificate", false, "")

	shimWritesFirst = flag.Bool("shim-writes-first", false, "")

	resumeCount = flag.Int("resume-count", 0, "")

	curves        = flagStringSlice("curves", "")
	expectedCurve = flag.String("expect-curve-id", "", "")

	shimID = flag.Uint64("shim-id", 0, "")
	_      = flag.Bool("ipv6", false, "")

	echConfigListB64           = flag.String("ech-config-list", "", "")
	expectECHAccepted          = flag.Bool("expect-ech-accept", false, "")
	expectHRR                  = flag.Bool("expect-hrr", false, "")
	expectNoHRR                = flag.Bool("expect-no-hrr", false, "")
	expectedECHRetryConfigs    = flag.String("expect-ech-retry-configs", "", "")
	expectNoECHRetryConfigs    = flag.Bool("expect-no-ech-retry-configs", false, "")
	onInitialExpectECHAccepted = flag.Bool("on-initial-expect-ech-accept", false, "")
	_                          = flag.Bool("expect-no-ech-name-override", false, "")
	_                          = flag.String("expect-ech-name-override", "", "")
	_                          = flag.Bool("reverify-on-resume", false, "")
	onResumeECHConfigListB64   = flag.String("on-resume-ech-config-list", "", "")
	_                          = flag.Bool("on-resume-expect-reject-early-data", false, "")
	onResumeExpectECHAccepted  = flag.Bool("on-resume-expect-ech-accept", false, "")
	_                          = flag.Bool("on-resume-expect-no-ech-name-override", false, "")
	expectedServerName         = flag.String("expect-server-name", "", "")
	echServerConfig            = flagStringSlice("ech-server-config", "")
	echServerKey               = flagStringSlice("ech-server-key", "")
	echServerRetryConfig       = flagStringSlice("ech-is-retry-config", "")

	expectSessionMiss = flag.Bool("expect-session-miss", false, "")

	_                       = flag.Bool("enable-early-data", false, "")
	_                       = flag.Bool("on-resume-expect-accept-early-data", false, "")
	_                       = flag.Bool("expect-ticket-supports-early-data", false, "")
	onResumeShimWritesFirst = flag.Bool("on-resume-shim-writes-first", false, "")

	advertiseALPN        = flag.String("advertise-alpn", "", "")
	expectALPN           = flag.String("expect-alpn", "", "")
	rejectALPN           = flag.Bool("reject-alpn", false, "")
	declineALPN          = flag.Bool("decline-alpn", false, "")
	expectAdvertisedALPN = flag.String("expect-advertised-alpn", "", "")
	selectALPN           = flag.String("select-alpn", "", "")

	hostName = flag.String("host-name", "", "")

	verifyPeer = flag.Bool("verify-peer", false, "")
	_          = flag.Bool("use-custom-verify-callback", false, "")
)

type stringSlice []string

func flagStringSlice(name, usage string) *stringSlice {
	f := &stringSlice{}
	flag.Var(f, name, usage)
	return f
}

func (saf *stringSlice) String() string {
	return strings.Join(*saf, ",")
}

func (saf *stringSlice) Set(s string) error {
	*saf = append(*saf, s)
	return nil
}

func bogoShim() {
	if *isHandshakerSupported {
		fmt.Println("No")
		return
	}

	cfg := &Config{
		ServerName: "test",

		MinVersion: uint16(*minVersion),
		MaxVersion: uint16(*maxVersion),

		ClientSessionCache: NewLRUClientSessionCache(0),

		GetConfigForClient: func(chi *ClientHelloInfo) (*Config, error) {

			if *expectAdvertisedALPN != "" {

				s := cryptobyte.String(*expectAdvertisedALPN)

				var expectedALPNs []string

				for !s.Empty() {
					var alpn cryptobyte.String
					if !s.ReadUint8LengthPrefixed(&alpn) {
						return nil, fmt.Errorf("unexpected error while parsing arguments for -expect-advertised-alpn")
					}
					expectedALPNs = append(expectedALPNs, string(alpn))
				}

				if !slices.Equal(chi.SupportedProtos, expectedALPNs) {
					return nil, fmt.Errorf("unexpected ALPN: got %q, want %q", chi.SupportedProtos, expectedALPNs)
				}
			}
			return nil, nil
		},
	}

	if *noTLS1 {
		cfg.MinVersion = VersionTLS11
		if *noTLS11 {
			cfg.MinVersion = VersionTLS12
			if *noTLS12 {
				cfg.MinVersion = VersionTLS13
				if *noTLS13 {
					log.Fatalf("no supported versions enabled")
				}
			}
		}
	} else if *noTLS13 {
		cfg.MaxVersion = VersionTLS12
		if *noTLS12 {
			cfg.MaxVersion = VersionTLS11
			if *noTLS11 {
				cfg.MaxVersion = VersionTLS10
				if *noTLS1 {
					log.Fatalf("no supported versions enabled")
				}
			}
		}
	}

	if *advertiseALPN != "" {
		alpns := *advertiseALPN
		for len(alpns) > 0 {
			alpnLen := int(alpns[0])
			cfg.NextProtos = append(cfg.NextProtos, alpns[1:1+alpnLen])
			alpns = alpns[alpnLen+1:]
		}
	}

	if *rejectALPN {
		cfg.NextProtos = []string{"unnegotiableprotocol"}
	}

	if *declineALPN {
		cfg.NextProtos = []string{}
	}
	if *selectALPN != "" {
		cfg.NextProtos = []string{*selectALPN}
	}

	if *hostName != "" {
		cfg.ServerName = *hostName
	}

	if *keyfile != "" || *certfile != "" {
		pair, err := LoadX509KeyPair(*certfile, *keyfile)
		if err != nil {
			log.Fatalf("load key-file err: %s", err)
		}
		cfg.Certificates = []Certificate{pair}
	}
	if *trustCert != "" {
		pool := x509.NewCertPool()
		certFile, err := os.ReadFile(*trustCert)
		if err != nil {
			log.Fatalf("load trust-cert err: %s", err)
		}
		block, _ := pem.Decode(certFile)
		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			log.Fatalf("parse trust-cert err: %s", err)
		}
		pool.AddCert(cert)
		cfg.RootCAs = pool
	}

	if *requireAnyClientCertificate {
		cfg.ClientAuth = RequireAnyClientCert
	}
	if *verifyPeer {
		cfg.ClientAuth = VerifyClientCertIfGiven
	}

	if *echConfigListB64 != "" {
		echConfigList, err := base64.StdEncoding.DecodeString(*echConfigListB64)
		if err != nil {
			log.Fatalf("parse ech-config-list err: %s", err)
		}
		cfg.EncryptedClientHelloConfigList = echConfigList
		cfg.MinVersion = VersionTLS13
	}

	if len(*curves) != 0 {
		for _, curveStr := range *curves {
			id, err := strconv.Atoi(curveStr)
			if err != nil {
				log.Fatalf("failed to parse curve id %q: %s", curveStr, err)
			}
			cfg.CurvePreferences = append(cfg.CurvePreferences, CurveID(id))
		}
	}

	if len(*echServerConfig) != 0 {
		if len(*echServerConfig) != len(*echServerKey) || len(*echServerConfig) != len(*echServerRetryConfig) {
			log.Fatal("-ech-server-config, -ech-server-key, and -ech-is-retry-config mismatch")
		}

		for i, c := range *echServerConfig {
			configBytes, err := base64.StdEncoding.DecodeString(c)
			if err != nil {
				log.Fatalf("parse ech-server-config err: %s", err)
			}
			privBytes, err := base64.StdEncoding.DecodeString((*echServerKey)[i])
			if err != nil {
				log.Fatalf("parse ech-server-key err: %s", err)
			}

			cfg.EncryptedClientHelloKeys = append(cfg.EncryptedClientHelloKeys, EncryptedClientHelloKey{
				Config:      configBytes,
				PrivateKey:  privBytes,
				SendAsRetry: (*echServerRetryConfig)[i] == "1",
			})
		}
	}

	for i := 0; i < *resumeCount+1; i++ {
		if i > 0 && (*onResumeECHConfigListB64 != "") {
			echConfigList, err := base64.StdEncoding.DecodeString(*onResumeECHConfigListB64)
			if err != nil {
				log.Fatalf("parse ech-config-list err: %s", err)
			}
			cfg.EncryptedClientHelloConfigList = echConfigList
		}

		conn, err := net.Dial("tcp", net.JoinHostPort("localhost", *port))
		if err != nil {
			log.Fatalf("dial err: %s", err)
		}
		defer conn.Close()

		// Write the shim ID we were passed as a little endian uint64
		shimIDBytes := make([]byte, 8)
		byteorder.LEPutUint64(shimIDBytes, *shimID)
		if _, err := conn.Write(shimIDBytes); err != nil {
			log.Fatalf("failed to write shim id: %s", err)
		}

		var tlsConn *Conn
		if *server {
			tlsConn = Server(conn, cfg)
		} else {
			tlsConn = Client(conn, cfg)
		}

		if i == 0 && *shimWritesFirst {
			if _, err := tlsConn.Write([]byte("hello")); err != nil {
				log.Fatalf("write err: %s", err)
			}
		}

		for {
			buf := make([]byte, 500)
			var n int
			n, err = tlsConn.Read(buf)
			if err != nil {
				break
			}
			buf = buf[:n]
			for i := range buf {
				buf[i] ^= 0xff
			}
			if _, err = tlsConn.Write(buf); err != nil {
				break
			}
		}
		if err != nil && err != io.EOF {
			retryErr, ok := err.(*ECHRejectionError)
			if !ok {
				log.Fatalf("unexpected error type returned: %v", err)
			}
			if *expectNoECHRetryConfigs && len(retryErr.RetryConfigList) > 0 {
				log.Fatalf("expected no ECH retry configs, got some")
			}
			if *expectedECHRetryConfigs != "" {
				expectedRetryConfigs, err := base64.StdEncoding.DecodeString(*expectedECHRetryConfigs)
				if err != nil {
					log.Fatalf("failed to decode expected retry configs: %s", err)
				}
				if !bytes.Equal(retryErr.RetryConfigList, expectedRetryConfigs) {
					log.Fatalf("unexpected retry list returned: got %x, want %x", retryErr.RetryConfigList, expectedRetryConfigs)
				}
			}
			log.Fatalf("conn error: %s", err)
		}

		cs := tlsConn.ConnectionState()
		if cs.HandshakeComplete {
			if *expectALPN != "" && cs.NegotiatedProtocol != *expectALPN {
				log.Fatalf("unexpected protocol negotiated: want %q, got %q", *expectALPN, cs.NegotiatedProtocol)
			}

			if *selectALPN != "" && cs.NegotiatedProtocol != *selectALPN {
				log.Fatalf("unexpected protocol negotiated: want %q, got %q", *selectALPN, cs.NegotiatedProtocol)
			}

			if *expectVersion != 0 && cs.Version != uint16(*expectVersion) {
				log.Fatalf("expected ssl version %q, got %q", uint16(*expectVersion), cs.Version)
			}
			if *declineALPN && cs.NegotiatedProtocol != "" {
				log.Fatal("unexpected ALPN protocol")
			}
			if *expectECHAccepted && !cs.ECHAccepted {
				log.Fatal("expected ECH to be accepted, but connection state shows it was not")
			} else if i == 0 && *onInitialExpectECHAccepted && !cs.ECHAccepted {
				log.Fatal("expected ECH to be accepted, but connection state shows it was not")
			} else if i > 0 && *onResumeExpectECHAccepted && !cs.ECHAccepted {
				log.Fatal("expected ECH to be accepted on resumption, but connection state shows it was not")
			} else if i == 0 && !*expectECHAccepted && cs.ECHAccepted {
				log.Fatal("did not expect ECH, but it was accepted")
			}

			if *expectHRR && !cs.testingOnlyDidHRR {
				log.Fatal("expected HRR but did not do it")
			}

			if *expectNoHRR && cs.testingOnlyDidHRR {
				log.Fatal("expected no HRR but did do it")
			}

			if *expectSessionMiss && cs.DidResume {
				log.Fatal("unexpected session resumption")
			}

			if *expectedServerName != "" && cs.ServerName != *expectedServerName {
				log.Fatalf("unexpected server name: got %q, want %q", cs.ServerName, *expectedServerName)
			}
		}

		if *expectedCurve != "" {
			expectedCurveID, err := strconv.Atoi(*expectedCurve)
			if err != nil {
				log.Fatalf("failed to parse -expect-curve-id: %s", err)
			}
			if tlsConn.curveID != CurveID(expectedCurveID) {
				log.Fatalf("unexpected curve id: want %d, got %d", expectedCurveID, tlsConn.curveID)
			}
		}
	}
}

func TestBogoSuite(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	if testenv.Builder() != "" && runtime.GOOS == "windows" {
		t.Skip("#66913: windows network connections are flakey on builders")
	}
	skipFIPS(t)

	// In order to make Go test caching work as expected, we stat the
	// bogo_config.json file, so that the Go testing hooks know that it is
	// important for this test and will invalidate a cached test result if the
	// file changes.
	if _, err := os.Stat("bogo_config.json"); err != nil {
		t.Fatal(err)
	}

	var bogoDir string
	if *bogoLocalDir != "" {
		bogoDir = *bogoLocalDir
	} else {
		const boringsslModVer = "v0.0.0-20241120195446-5cce3fbd23e1"
		bogoDir = cryptotest.FetchModule(t, "boringssl.googlesource.com/boringssl.git", boringsslModVer)
	}

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	resultsFile := filepath.Join(t.TempDir(), "results.json")

	args := []string{
		"test",
		".",
		fmt.Sprintf("-shim-config=%s", filepath.Join(cwd, "bogo_config.json")),
		fmt.Sprintf("-shim-path=%s", os.Args[0]),
		"-shim-extra-flags=-bogo-mode",
		"-allow-unimplemented",
		"-loose-errors", // TODO(roland): this should be removed eventually
		fmt.Sprintf("-json-output=%s", resultsFile),
	}
	if *bogoFilter != "" {
		args = append(args, fmt.Sprintf("-test=%s", *bogoFilter))
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
	out := &strings.Builder{}
	cmd.Stderr = out
	cmd.Dir = filepath.Join(bogoDir, "ssl/test/runner")
	err = cmd.Run()
	// NOTE: we don't immediately check the error, because the failure could be either because
	// the runner failed for some unexpected reason, or because a test case failed, and we
	// cannot easily differentiate these cases. We check if the JSON results file was written,
	// which should only happen if the failure was because of a test failure, and use that
	// to determine the failure mode.

	resultsJSON, jsonErr := os.ReadFile(resultsFile)
	if jsonErr != nil {
		if err != nil {
			t.Fatalf("bogo failed: %s\n%s", err, out)
		}
		t.Fatalf("failed to read results JSON file: %s", jsonErr)
	}

	var results bogoResults
	if err := json.Unmarshal(resultsJSON, &results); err != nil {
		t.Fatalf("failed to parse results JSON: %s", err)
	}

	// assertResults contains test results we want to make sure
	// are present in the output. They are only checked if -bogo-filter
	// was not passed.
	assertResults := map[string]string{
		"CurveTest-Client-MLKEM-TLS13": "PASS",
		"CurveTest-Server-MLKEM-TLS13": "PASS",
	}

	for name, result := range results.Tests {
		// This is not really the intended way to do this... but... it works?
		t.Run(name, func(t *testing.T) {
			if result.Actual == "FAIL" && result.IsUnexpected {
				t.Fatal(result.Error)
			}
			if expectedResult, ok := assertResults[name]; ok && expectedResult != result.Actual {
				t.Fatalf("unexpected result: got %s, want %s", result.Actual, assertResults[name])
			}
			delete(assertResults, name)
			if result.Actual == "SKIP" {
				t.Skip()
			}
		})
	}
	if *bogoFilter == "" {
		// Anything still in assertResults did not show up in the results, so we should fail
		for name, expectedResult := range assertResults {
			t.Run(name, func(t *testing.T) {
				t.Fatalf("expected test to run with result %s, but it was not present in the test results", expectedResult)
			})
		}
	}
}

// bogoResults is a copy of boringssl.googlesource.com/boringssl/testresults.Results
type bogoResults struct {
	Version           int            `json:"version"`
	Interrupted       bool           `json:"interrupted"`
	PathDelimiter     string         `json:"path_delimiter"`
	SecondsSinceEpoch float64        `json:"seconds_since_epoch"`
	NumFailuresByType map[string]int `json:"num_failures_by_type"`
	Tests             map[string]struct {
		Actual       string `json:"actual"`
		Expected     string `json:"expected"`
		IsUnexpected bool   `json:"is_unexpected"`
		Error        string `json:"error,omitempty"`
	} `json:"tests"`
}
