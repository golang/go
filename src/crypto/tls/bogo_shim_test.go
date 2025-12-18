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
	"errors"
	"flag"
	"fmt"
	"html/template"
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
	"time"

	"golang.org/x/crypto/cryptobyte"
)

const boringsslModVer = "v0.0.0-20250620172916-f51d8b099832"

var (
	port   = flag.String("port", "", "")
	server = flag.Bool("server", false, "")

	isHandshakerSupported = flag.Bool("is-handshaker-supported", false, "")

	keyfile      = flag.String("key-file", "", "")
	certfile     = flag.String("cert-file", "", "")
	ocspResponse = flagBase64("ocsp-response", "")
	signingPrefs = flagIntSlice("signing-prefs", "")

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

	curves        = flagIntSlice("curves", "")
	expectedCurve = flag.String("expect-curve-id", "", "")

	verifyPrefs        = flagIntSlice("verify-prefs", "")
	expectedSigAlg     = flag.String("expect-peer-signature-algorithm", "", "")
	expectedPeerSigAlg = flagIntSlice("expect-peer-verify-pref", "")

	shimID = flag.Uint64("shim-id", 0, "")
	_      = flag.Bool("ipv6", false, "")

	echConfigList              = flagBase64("ech-config-list", "")
	expectECHAccepted          = flag.Bool("expect-ech-accept", false, "")
	expectHRR                  = flag.Bool("expect-hrr", false, "")
	expectNoHRR                = flag.Bool("expect-no-hrr", false, "")
	expectedECHRetryConfigs    = flag.String("expect-ech-retry-configs", "", "")
	expectNoECHRetryConfigs    = flag.Bool("expect-no-ech-retry-configs", false, "")
	onInitialExpectECHAccepted = flag.Bool("on-initial-expect-ech-accept", false, "")
	_                          = flag.Bool("expect-no-ech-name-override", false, "")
	_                          = flag.String("expect-ech-name-override", "", "")
	_                          = flag.Bool("reverify-on-resume", false, "")
	onResumeECHConfigList      = flagBase64("on-resume-ech-config-list", "")
	_                          = flag.Bool("on-resume-expect-reject-early-data", false, "")
	onResumeExpectECHAccepted  = flag.Bool("on-resume-expect-ech-accept", false, "")
	_                          = flag.Bool("on-resume-expect-no-ech-name-override", false, "")
	expectedServerName         = flag.String("expect-server-name", "", "")
	echServerConfig            = flagStringSlice("ech-server-config", "")
	echServerKey               = flagStringSlice("ech-server-key", "")
	echServerRetryConfig       = flagStringSlice("ech-is-retry-config", "")

	expectSessionMiss = flag.Bool("expect-session-miss", false, "")

	_ = flag.Bool("enable-early-data", false, "")
	_ = flag.Bool("on-resume-expect-accept-early-data", false, "")
	_ = flag.Bool("expect-ticket-supports-early-data", false, "")
	_ = flag.Bool("on-resume-shim-writes-first", false, "")

	advertiseALPN        = flag.String("advertise-alpn", "", "")
	expectALPN           = flag.String("expect-alpn", "", "")
	rejectALPN           = flag.Bool("reject-alpn", false, "")
	declineALPN          = flag.Bool("decline-alpn", false, "")
	expectAdvertisedALPN = flag.String("expect-advertised-alpn", "", "")
	selectALPN           = flag.String("select-alpn", "", "")

	hostName = flag.String("host-name", "", "")

	verifyPeer = flag.Bool("verify-peer", false, "")
	_          = flag.Bool("use-custom-verify-callback", false, "")

	waitForDebugger = flag.Bool("wait-for-debugger", false, "")
)

type stringSlice []string

func flagStringSlice(name, usage string) *stringSlice {
	f := new(stringSlice)
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

type intSlice []int64

func flagIntSlice(name, usage string) *intSlice {
	f := new(intSlice)
	flag.Var(f, name, usage)
	return f
}

func (sf *intSlice) String() string {
	return strings.Join(strings.Split(fmt.Sprint(*sf), " "), ",")
}

func (sf *intSlice) Set(s string) error {
	i, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return err
	}
	*sf = append(*sf, i)
	return nil
}

type base64Flag []byte

func flagBase64(name, usage string) *base64Flag {
	f := new(base64Flag)
	flag.Var(f, name, usage)
	return f
}

func (f *base64Flag) String() string {
	return base64.StdEncoding.EncodeToString(*f)
}

func (f *base64Flag) Set(s string) error {
	if *f != nil {
		return fmt.Errorf("multiple base64 values not supported")
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return err
	}
	*f = b
	return nil
}

func bogoShim() {
	if *isHandshakerSupported {
		fmt.Println("No")
		return
	}

	fmt.Printf("BoGo shim flags: %q", os.Args[1:])

	// Test with both the default and insecure cipher suites.
	var ciphersuites []uint16
	for _, s := range append(CipherSuites(), InsecureCipherSuites()...) {
		ciphersuites = append(ciphersuites, s.ID)
	}

	cfg := &Config{
		ServerName: "test",

		MinVersion: uint16(*minVersion),
		MaxVersion: uint16(*maxVersion),

		ClientSessionCache: NewLRUClientSessionCache(0),

		CipherSuites: ciphersuites,

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
		for _, id := range *signingPrefs {
			pair.SupportedSignatureAlgorithms = append(pair.SupportedSignatureAlgorithms, SignatureScheme(id))
		}
		pair.OCSPStaple = *ocspResponse
		// Use Get[Client]Certificate to force the use of the certificate, which
		// more closely matches the BoGo expectations (e.g. handshake failure if
		// no client certificates are compatible).
		cfg.GetCertificate = func(chi *ClientHelloInfo) (*Certificate, error) {
			if *expectedPeerSigAlg != nil {
				if len(chi.SignatureSchemes) != len(*expectedPeerSigAlg) {
					return nil, fmt.Errorf("unexpected signature algorithms: got %s, want %v", chi.SignatureSchemes, *expectedPeerSigAlg)
				}
				for i := range *expectedPeerSigAlg {
					if chi.SignatureSchemes[i] != SignatureScheme((*expectedPeerSigAlg)[i]) {
						return nil, fmt.Errorf("unexpected signature algorithms: got %s, want %v", chi.SignatureSchemes, *expectedPeerSigAlg)
					}
				}
			}
			return &pair, nil
		}
		cfg.GetClientCertificate = func(cri *CertificateRequestInfo) (*Certificate, error) {
			if *expectedPeerSigAlg != nil {
				if len(cri.SignatureSchemes) != len(*expectedPeerSigAlg) {
					return nil, fmt.Errorf("unexpected signature algorithms: got %s, want %v", cri.SignatureSchemes, *expectedPeerSigAlg)
				}
				for i := range *expectedPeerSigAlg {
					if cri.SignatureSchemes[i] != SignatureScheme((*expectedPeerSigAlg)[i]) {
						return nil, fmt.Errorf("unexpected signature algorithms: got %s, want %v", cri.SignatureSchemes, *expectedPeerSigAlg)
					}
				}
			}
			return &pair, nil
		}
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

	if *echConfigList != nil {
		cfg.EncryptedClientHelloConfigList = *echConfigList
		cfg.MinVersion = VersionTLS13
	}

	if *curves != nil {
		for _, id := range *curves {
			cfg.CurvePreferences = append(cfg.CurvePreferences, CurveID(id))
		}
	}

	if *verifyPrefs != nil {
		for _, id := range *verifyPrefs {
			testingOnlySupportedSignatureAlgorithms = append(testingOnlySupportedSignatureAlgorithms, SignatureScheme(id))
		}
	}

	if *echServerConfig != nil {
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
		if i > 0 && *onResumeECHConfigList != nil {
			cfg.EncryptedClientHelloConfigList = *onResumeECHConfigList
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

		// If we were instructed to wait for a debugger, then send SIGSTOP to ourselves.
		// When the debugger attaches it will continue the process.
		if *waitForDebugger {
			pauseProcess()
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
		if err != io.EOF {
			// Flush the TLS conn and then perform a graceful shutdown of the
			// TCP connection to avoid the runner side hitting an unexpected
			// write error before it has processed the alert we may have
			// generated for the error condition.
			orderlyShutdown(tlsConn)

			retryErr, ok := err.(*ECHRejectionError)
			if !ok {
				log.Fatal(err)
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
				log.Fatalf("expected ssl version %d, got %d", *expectVersion, cs.Version)
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

			if *expectHRR && !cs.HelloRetryRequest {
				log.Fatal("expected HRR but did not do it")
			}

			if *expectNoHRR && cs.HelloRetryRequest {
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
			if cs.CurveID != CurveID(expectedCurveID) {
				log.Fatalf("unexpected curve id: want %d, got %d", expectedCurveID, tlsConn.curveID)
			}
		}

		// TODO: implement testingOnlyPeerSignatureAlgorithm on resumption.
		if *expectedSigAlg != "" && !cs.DidResume {
			expectedSigAlgID, err := strconv.Atoi(*expectedSigAlg)
			if err != nil {
				log.Fatalf("failed to parse -expect-peer-signature-algorithm: %s", err)
			}
			if cs.testingOnlyPeerSignatureAlgorithm != SignatureScheme(expectedSigAlgID) {
				log.Fatalf("unexpected peer signature algorithm: want %s, got %s", SignatureScheme(expectedSigAlgID), cs.testingOnlyPeerSignatureAlgorithm)
			}
		}
	}
}

// If the test case produces an error, we don't want to immediately close the
// TCP connection after generating an alert. The runner side may try to write
// additional data to the connection before it reads the alert. If the conn
// has already been torn down, then these writes will produce an unexpected
// broken pipe err and fail the test.
func orderlyShutdown(tlsConn *Conn) {
	// Flush any pending alert data
	tlsConn.flush()

	netConn := tlsConn.NetConn()
	tcpConn := netConn.(*net.TCPConn)
	tcpConn.CloseWrite()

	// Read and discard any data that was sent by the peer.
	buf := make([]byte, maxPlaintext)
	for {
		n, err := tcpConn.Read(buf)
		if n == 0 || err != nil {
			break
		}
	}

	tcpConn.CloseRead()
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
		ensureLocalBogo(t, *bogoLocalDir)
		bogoDir = *bogoLocalDir
	} else {
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
	cmd.Dir = filepath.Join(bogoDir, "ssl/test/runner")
	out, err := cmd.CombinedOutput()
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

	if *bogoReport != "" {
		if err := generateReport(results, *bogoReport); err != nil {
			t.Fatalf("failed to generate report: %v", err)
		}
	}

	// assertResults contains test results we want to make sure
	// are present in the output. They are only checked if -bogo-filter
	// was not passed.
	assertResults := map[string]string{
		"CurveTest-Client-MLKEM-TLS13": "PASS",
		"CurveTest-Server-MLKEM-TLS13": "PASS",

		// Various signature algorithm tests checking that we enforce our
		// preferences on the peer.
		"ClientAuth-Enforced":                    "PASS",
		"ServerAuth-Enforced":                    "PASS",
		"ClientAuth-Enforced-TLS13":              "PASS",
		"ServerAuth-Enforced-TLS13":              "PASS",
		"VerifyPreferences-Advertised":           "PASS",
		"VerifyPreferences-Enforced":             "PASS",
		"Client-TLS12-NoSign-RSA_PKCS1_MD5_SHA1": "PASS",
		"Server-TLS12-NoSign-RSA_PKCS1_MD5_SHA1": "PASS",
		"Client-TLS13-NoSign-RSA_PKCS1_MD5_SHA1": "PASS",
		"Server-TLS13-NoSign-RSA_PKCS1_MD5_SHA1": "PASS",
	}

	for name, result := range results.Tests {
		// This is not really the intended way to do this... but... it works?
		t.Run(name, func(t *testing.T) {
			if result.Actual == "FAIL" && result.IsUnexpected {
				t.Fail()
			}
			if result.Error != "" {
				t.Log(result.Error)
			}
			if exp, ok := assertResults[name]; ok && exp != result.Actual {
				t.Errorf("unexpected result: got %s, want %s", result.Actual, exp)
			}
			delete(assertResults, name)
			if result.Actual == "SKIP" {
				t.SkipNow()
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

// ensureLocalBogo fetches BoringSSL to localBogoDir at the correct revision
// (from boringsslModVer) if localBogoDir doesn't already exist.
//
// If localBogoDir does exist, ensureLocalBogo fails the test if it isn't
// a directory.
func ensureLocalBogo(t *testing.T, localBogoDir string) {
	t.Helper()

	if stat, err := os.Stat(localBogoDir); err == nil {
		if !stat.IsDir() {
			t.Fatalf("local bogo dir (%q) exists but is not a directory", localBogoDir)
		}

		t.Logf("using local bogo checkout from %q", localBogoDir)
		return
	} else if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed to stat local bogo dir (%q): %v", localBogoDir, err)
	}

	testenv.MustHaveExecPath(t, "git")

	idx := strings.LastIndex(boringsslModVer, "-")
	if idx == -1 || idx == len(boringsslModVer)-1 {
		t.Fatalf("invalid boringsslModVer format: %q", boringsslModVer)
	}
	commitSHA := boringsslModVer[idx+1:]

	t.Logf("cloning boringssl@%s to %q", commitSHA, localBogoDir)
	cloneCmd := testenv.Command(t, "git", "clone", "--no-checkout", "https://boringssl.googlesource.com/boringssl", localBogoDir)
	if err := cloneCmd.Run(); err != nil {
		t.Fatalf("git clone failed: %v", err)
	}

	checkoutCmd := testenv.Command(t, "git", "checkout", commitSHA)
	checkoutCmd.Dir = localBogoDir
	if err := checkoutCmd.Run(); err != nil {
		t.Fatalf("git checkout failed: %v", err)
	}

	t.Logf("using fresh local bogo checkout from %q", localBogoDir)
}

func generateReport(results bogoResults, outPath string) error {
	data := reportData{
		Results:   results,
		Timestamp: time.Unix(int64(results.SecondsSinceEpoch), 0).Format("2006-01-02 15:04:05"),
		Revision:  boringsslModVer,
	}

	tmpl := template.Must(template.New("report").Parse(reportTemplate))
	file, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer file.Close()

	return tmpl.Execute(file, data)
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

type reportData struct {
	Results     bogoResults
	SkipReasons map[string]string
	Timestamp   string
	Revision    string
}

const reportTemplate = `
<!DOCTYPE html>
<html>
<head>
    <title>BoGo Results Report</title>
    <style>
        body { font-family: monospace; margin: 20px; }
        .summary { background: #f5f5f5; padding: 10px; margin-bottom: 20px; }
        .controls { margin-bottom: 10px; }
        .controls input, select { margin-right: 10px; }
        table { width: 100%; border-collapse: collapse; table-layout: fixed; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
        th { background-color: #f2f2f2; cursor: pointer; }
        .name-col { width: 30%; }
        .status-col { width: 8%; }
        .actual-col { width: 8%; }
        .expected-col { width: 8%; }
        .error-col { width: 26%; }
        .PASS { background-color: #d4edda; }
        .FAIL { background-color: #f8d7da; }
        .SKIP { background-color: #fff3cd; }
        .error {
            font-family: monospace;
            font-size: 0.9em;
            color: #721c24;
            white-space: pre-wrap;
            word-break: break-word;
        }
    </style>
</head>
<body>
<h1>BoGo Results Report</h1>

<div class="summary">
    <strong>Generated:</strong> {{.Timestamp}} | <strong>BoGo Revision:</strong> {{.Revision}}<br>
    {{range $status, $count := .Results.NumFailuresByType}}
    <strong>{{$status}}:</strong> {{$count}} |
    {{end}}
</div>

<div class="controls">
    <input type="text" id="search" placeholder="Search tests..." onkeyup="filterTests()">
    <select id="statusFilter" onchange="filterTests()">
        <option value="">All</option>
        <option value="FAIL">Failed</option>
        <option value="PASS">Passed</option>
        <option value="SKIP">Skipped</option>
    </select>
</div>

<table id="resultsTable">
    <thead>
    <tr>
        <th class="name-col" onclick="sortBy('name')">Test Name</th>
        <th class="status-col" onclick="sortBy('status')">Status</th>
        <th class="actual-col" onclick="sortBy('actual')">Actual</th>
        <th class="expected-col" onclick="sortBy('expected')">Expected</th>
        <th class="error-col">Error</th>
    </tr>
    </thead>
    <tbody>
    {{range $name, $test := .Results.Tests}}
    <tr class="{{$test.Actual}}" data-name="{{$name}}" data-status="{{$test.Actual}}">
        <td>{{$name}}</td>
        <td>{{$test.Actual}}</td>
        <td>{{$test.Actual}}</td>
        <td>{{$test.Expected}}</td>
        <td class="error">{{$test.Error}}</td>
    </tr>
    {{end}}
    </tbody>
</table>

<script>
    function filterTests() {
        const search = document.getElementById('search').value.toLowerCase();
        const status = document.getElementById('statusFilter').value;
        const rows = document.querySelectorAll('#resultsTable tbody tr');

        rows.forEach(row => {
            const name = row.dataset.name.toLowerCase();
            const rowStatus = row.dataset.status;
            const matchesSearch = name.includes(search);
            const matchesStatus = !status || rowStatus === status;

            row.style.display = matchesSearch && matchesStatus ? '' : 'none';
        });
    }

    function sortBy(column) {
        const tbody = document.querySelector('#resultsTable tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));

        rows.sort((a, b) => {
            if (column === 'status') {
                const statusOrder = {'FAIL': 0, 'PASS': 1, 'SKIP': 2};
                const aStatus = a.dataset.status;
                const bStatus = b.dataset.status;
                if (aStatus !== bStatus) {
                    return statusOrder[aStatus] - statusOrder[bStatus];
                }
                return a.dataset.name.localeCompare(b.dataset.name);
            } else {
                return a.dataset.name.localeCompare(b.dataset.name);
            }
        });

        rows.forEach(row => tbody.appendChild(row));
        filterTests();
    }

    sortBy("status");
</script>
</body>
</html>
`
