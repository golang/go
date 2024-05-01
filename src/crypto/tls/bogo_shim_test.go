package tls

import (
	"crypto/x509"
	"encoding/binary"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"internal/testenv"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

var (
	port   = flag.String("port", "", "")
	server = flag.Bool("server", false, "")

	isHandshakerSupported = flag.Bool("is-handshaker-supported", false, "")

	keyfile  = flag.String("key-file", "", "")
	certfile = flag.String("cert-file", "", "")

	trustCert = flag.String("trust-cert", "", "")

	minVersion = flag.Int("min-version", VersionSSL30, "")
	maxVersion = flag.Int("max-version", VersionTLS13, "")

	noTLS13 = flag.Bool("no-tls13", false, "")

	requireAnyClientCertificate = flag.Bool("require-any-client-certificate", false, "")

	shimWritesFirst = flag.Bool("shim-writes-first", false, "")

	resumeCount = flag.Int("resume-count", 0, "")

	shimID = flag.Uint64("shim-id", 0, "")
	_      = flag.Bool("ipv6", false, "")

	// Unimplemented flags
	// -advertise-alpn
	// -advertise-npn
	// -allow-hint-mismatch
	// -async
	// -check-close-notify
	// -cipher
	// -curves
	// -delegated-credential
	// -dtls
	// -ech-config-list
	// -ech-server-config
	// -enable-channel-id
	// -enable-early-data
	// -enable-ech-grease
	// -enable-grease
	// -enable-ocsp-stapling
	// -enable-signed-cert-timestamps
	// -expect-advertised-alpn
	// -expect-certificate-types
	// -expect-channel-id
	// -expect-cipher-aes
	// -expect-client-ca-list
	// -expect-curve-id
	// -expect-early-data-reason
	// -expect-extended-master-secret
	// -expect-hrr
	// -expect-key-usage-invalid
	// -expect-msg-callback
	// -expect-no-session
	// -expect-peer-cert-file
	// -expect-peer-signature-algorithm
	// -expect-peer-verify-pref
	// -expect-secure-renegotiation
	// -expect-server-name
	// -expect-ticket-supports-early-data
	// -export-keying-material
	// -export-traffic-secrets
	// -fail-cert-callback
	// -fail-early-callback
	// -fallback-scsv
	// -false-start
	// -forbid-renegotiation-after-handshake
	// -handshake-twice
	// -host-name
	// -ignore-rsa-key-usage
	// -implicit-handshake
	// -install-cert-compression-algs
	// -install-ddos-callback
	// -install-one-cert-compression-alg
	// -jdk11-workaround
	// -key-update
	// -max-cert-list
	// -max-send-fragment
	// -no-ticket
	// -no-tls1
	// -no-tls11
	// -no-tls12
	// -ocsp-response
	// -on-resume-expect-accept-early-data
	// -on-resume-expect-reject-early-data
	// -on-shim-cipher
	// -on-shim-curves
	// -peek-then-read
	// -psk
	// -read-with-unfinished-write
	// -reject-alpn
	// -renegotiate-explicit
	// -renegotiate-freely
	// -renegotiate-ignore
	// -renegotiate-once
	// -select-alpn
	// -select-next-proto
	// -send-alert
	// -send-channel-id
	// -server-preference
	// -shim-shuts-down
	// -signed-cert-timestamps
	// -signing-prefs
	// -srtp-profiles
	// -tls-unique
	// -use-client-ca-list
	// -use-ocsp-callback
	// -use-old-client-cert-callback
	// -verify-fail
	// -verify-peer
	// -verify-prefs
)

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
	}

	if *noTLS13 && cfg.MaxVersion == VersionTLS13 {
		cfg.MaxVersion = VersionTLS12
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

	for i := 0; i < *resumeCount+1; i++ {
		conn, err := net.Dial("tcp", net.JoinHostPort("localhost", *port))
		if err != nil {
			log.Fatalf("dial err: %s", err)
		}
		defer conn.Close()

		// Write the shim ID we were passed as a little endian uint64
		shimIDBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(shimIDBytes, *shimID)
		if _, err := conn.Write(shimIDBytes); err != nil {
			log.Fatalf("failed to write shim id: %s", err)
		}

		var tlsConn *Conn
		if *server {
			tlsConn = Server(conn, cfg)
		} else {
			tlsConn = Client(conn, cfg)
		}

		if *shimWritesFirst {
			if _, err := tlsConn.Write([]byte("hello")); err != nil {
				log.Fatalf("write err: %s", err)
			}
		}

		for {
			buf := make([]byte, 500)
			n, err := tlsConn.Read(buf)
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatalf("read err: %s", err)
			}
			buf = buf[:n]
			for i := range buf {
				buf[i] ^= 0xff
			}
			if _, err := tlsConn.Write(buf); err != nil {
				log.Fatalf("write err: %s", err)
			}
		}
	}
}

func TestBogoSuite(t *testing.T) {
	testenv.SkipIfShortAndSlow(t)
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveGoRun(t)
	testenv.MustHaveExec(t)

	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	if testenv.Builder() != "" && runtime.GOOS == "windows" {
		t.Skip("#66913: windows network connections are flakey on builders")
	}

	const boringsslModVer = "v0.0.0-20240412155355-1c6e10495e4f"
	output, err := exec.Command("go", "mod", "download", "-json", "github.com/google/boringssl@"+boringsslModVer).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to download boringssl: %s", err)
	}
	var j struct {
		Dir string
	}
	if err := json.Unmarshal(output, &j); err != nil {
		t.Fatalf("failed to parse 'go mod download' output: %s", err)
	}

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	args := []string{
		"test",
		".",
		fmt.Sprintf("-shim-config=%s", filepath.Join(cwd, "bogo_config.json")),
		fmt.Sprintf("-shim-path=%s", os.Args[0]),
		"-shim-extra-flags=-bogo-mode",
		"-allow-unimplemented",
		"-loose-errors", // TODO(roland): this should be removed eventually
	}
	if *bogoFilter != "" {
		args = append(args, fmt.Sprintf("-test=%s", *bogoFilter))
	}

	goCmd, err := testenv.GoTool()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(goCmd, args...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	cmd.Dir = filepath.Join(j.Dir, "ssl/test/runner")
	err = cmd.Run()
	if err != nil {
		t.Fatalf("bogo failed: %s", err)
	}
}
