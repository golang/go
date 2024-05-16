// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bufio"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// TLS reference tests run a connection against a reference implementation
// (OpenSSL) of TLS and record the bytes of the resulting connection. The Go
// code, during a test, is configured with deterministic randomness and so the
// reference test can be reproduced exactly in the future.
//
// In order to save everyone who wishes to run the tests from needing the
// reference implementation installed, the reference connections are saved in
// files in the testdata directory. Thus running the tests involves nothing
// external, but creating and updating them requires the reference
// implementation.
//
// Tests can be updated by running them with the -update flag. This will cause
// the test files for failing tests to be regenerated. Since the reference
// implementation will always generate fresh random numbers, large parts of the
// reference connection will always change.

var (
	update     = flag.Bool("update", false, "update golden files on failure")
	fast       = flag.Bool("fast", false, "impose a quick, possibly flaky timeout on recorded tests")
	keyFile    = flag.String("keylog", "", "destination file for KeyLogWriter")
	bogoMode   = flag.Bool("bogo-mode", false, "Enabled bogo shim mode, ignore everything else")
	bogoFilter = flag.String("bogo-filter", "", "BoGo test filter")
)

func runTestAndUpdateIfNeeded(t *testing.T, name string, run func(t *testing.T, update bool), wait bool) {
	success := t.Run(name, func(t *testing.T) {
		if !*update && !wait {
			t.Parallel()
		}
		run(t, false)
	})

	if !success && *update {
		t.Run(name+"#update", func(t *testing.T) {
			run(t, true)
		})
	}
}

// checkOpenSSLVersion ensures that the version of OpenSSL looks reasonable
// before updating the test data.
func checkOpenSSLVersion() error {
	if !*update {
		return nil
	}

	openssl := exec.Command("openssl", "version")
	output, err := openssl.CombinedOutput()
	if err != nil {
		return err
	}

	version := string(output)
	if strings.HasPrefix(version, "OpenSSL 1.1.1") {
		return nil
	}

	println("***********************************************")
	println("")
	println("You need to build OpenSSL 1.1.1 from source in order")
	println("to update the test data.")
	println("")
	println("Configure it with:")
	println("./Configure enable-weak-ssl-ciphers no-shared")
	println("and then add the apps/ directory at the front of your PATH.")
	println("***********************************************")

	return errors.New("version of OpenSSL does not appear to be suitable for updating test data")
}

// recordingConn is a net.Conn that records the traffic that passes through it.
// WriteTo can be used to produce output that can be later be loaded with
// ParseTestData.
type recordingConn struct {
	net.Conn
	sync.Mutex
	flows   [][]byte
	reading bool
}

func (r *recordingConn) Read(b []byte) (n int, err error) {
	if n, err = r.Conn.Read(b); n == 0 {
		return
	}
	b = b[:n]

	r.Lock()
	defer r.Unlock()

	if l := len(r.flows); l == 0 || !r.reading {
		buf := make([]byte, len(b))
		copy(buf, b)
		r.flows = append(r.flows, buf)
	} else {
		r.flows[l-1] = append(r.flows[l-1], b[:n]...)
	}
	r.reading = true
	return
}

func (r *recordingConn) Write(b []byte) (n int, err error) {
	if n, err = r.Conn.Write(b); n == 0 {
		return
	}
	b = b[:n]

	r.Lock()
	defer r.Unlock()

	if l := len(r.flows); l == 0 || r.reading {
		buf := make([]byte, len(b))
		copy(buf, b)
		r.flows = append(r.flows, buf)
	} else {
		r.flows[l-1] = append(r.flows[l-1], b[:n]...)
	}
	r.reading = false
	return
}

// WriteTo writes Go source code to w that contains the recorded traffic.
func (r *recordingConn) WriteTo(w io.Writer) (int64, error) {
	// TLS always starts with a client to server flow.
	clientToServer := true
	var written int64
	for i, flow := range r.flows {
		source, dest := "client", "server"
		if !clientToServer {
			source, dest = dest, source
		}
		n, err := fmt.Fprintf(w, ">>> Flow %d (%s to %s)\n", i+1, source, dest)
		written += int64(n)
		if err != nil {
			return written, err
		}
		dumper := hex.Dumper(w)
		n, err = dumper.Write(flow)
		written += int64(n)
		if err != nil {
			return written, err
		}
		err = dumper.Close()
		if err != nil {
			return written, err
		}
		clientToServer = !clientToServer
	}
	return written, nil
}

func parseTestData(r io.Reader) (flows [][]byte, err error) {
	var currentFlow []byte

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		// If the line starts with ">>> " then it marks the beginning
		// of a new flow.
		if strings.HasPrefix(line, ">>> ") {
			if len(currentFlow) > 0 || len(flows) > 0 {
				flows = append(flows, currentFlow)
				currentFlow = nil
			}
			continue
		}

		// Otherwise the line is a line of hex dump that looks like:
		// 00000170  fc f5 06 bf (...)  |.....X{&?......!|
		// (Some bytes have been omitted from the middle section.)
		_, after, ok := strings.Cut(line, " ")
		if !ok {
			return nil, errors.New("invalid test data")
		}
		line = after

		before, _, ok := strings.Cut(line, "|")
		if !ok {
			return nil, errors.New("invalid test data")
		}
		line = before

		hexBytes := strings.Fields(line)
		for _, hexByte := range hexBytes {
			val, err := strconv.ParseUint(hexByte, 16, 8)
			if err != nil {
				return nil, errors.New("invalid hex byte in test data: " + err.Error())
			}
			currentFlow = append(currentFlow, byte(val))
		}
	}

	if len(currentFlow) > 0 {
		flows = append(flows, currentFlow)
	}

	return flows, nil
}

// tempFile creates a temp file containing contents and returns its path.
func tempFile(contents string) string {
	file, err := os.CreateTemp("", "go-tls-test")
	if err != nil {
		panic("failed to create temp file: " + err.Error())
	}
	path := file.Name()
	file.WriteString(contents)
	file.Close()
	return path
}

// localListener is set up by TestMain and used by localPipe to create Conn
// pairs like net.Pipe, but connected by an actual buffered TCP connection.
var localListener struct {
	mu   sync.Mutex
	addr net.Addr
	ch   chan net.Conn
}

const localFlakes = 0 // change to 1 or 2 to exercise localServer/localPipe handling of mismatches

func localServer(l net.Listener) {
	for n := 0; ; n++ {
		c, err := l.Accept()
		if err != nil {
			return
		}
		if localFlakes == 1 && n%2 == 0 {
			c.Close()
			continue
		}
		localListener.ch <- c
	}
}

var isConnRefused = func(err error) bool { return false }

func localPipe(t testing.TB) (net.Conn, net.Conn) {
	localListener.mu.Lock()
	defer localListener.mu.Unlock()

	addr := localListener.addr

	var err error
Dialing:
	// We expect a rare mismatch, but probably not 5 in a row.
	for i := 0; i < 5; i++ {
		tooSlow := time.NewTimer(1 * time.Second)
		defer tooSlow.Stop()
		var c1 net.Conn
		c1, err = net.Dial(addr.Network(), addr.String())
		if err != nil {
			if runtime.GOOS == "dragonfly" && (isConnRefused(err) || os.IsTimeout(err)) {
				// golang.org/issue/29583: Dragonfly sometimes returns a spurious
				// ECONNREFUSED or ETIMEDOUT.
				<-tooSlow.C
				continue
			}
			t.Fatalf("localPipe: %v", err)
		}
		if localFlakes == 2 && i == 0 {
			c1.Close()
			continue
		}
		for {
			select {
			case <-tooSlow.C:
				t.Logf("localPipe: timeout waiting for %v", c1.LocalAddr())
				c1.Close()
				continue Dialing

			case c2 := <-localListener.ch:
				if c2.RemoteAddr().String() == c1.LocalAddr().String() {
					return c1, c2
				}
				t.Logf("localPipe: unexpected connection: %v != %v", c2.RemoteAddr(), c1.LocalAddr())
				c2.Close()
			}
		}
	}

	t.Fatalf("localPipe: failed to connect: %v", err)
	panic("unreachable")
}

// zeroSource is an io.Reader that returns an unlimited number of zero bytes.
type zeroSource struct{}

func (zeroSource) Read(b []byte) (n int, err error) {
	clear(b)
	return len(b), nil
}

func allCipherSuites() []uint16 {
	ids := make([]uint16, len(cipherSuites))
	for i, suite := range cipherSuites {
		ids[i] = suite.id
	}

	return ids
}

var testConfig *Config

func TestMain(m *testing.M) {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage of %s:\n", os.Args)
		flag.PrintDefaults()
		if *bogoMode {
			os.Exit(89)
		}
	}

	flag.Parse()

	if *bogoMode {
		bogoShim()
		os.Exit(0)
	}

	os.Exit(runMain(m))
}

func runMain(m *testing.M) int {
	// Cipher suites preferences change based on the architecture. Force them to
	// the version without AES acceleration for test consistency.
	hasAESGCMHardwareSupport = false

	// Set up localPipe.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		l, err = net.Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open local listener: %v", err)
		os.Exit(1)
	}
	localListener.ch = make(chan net.Conn)
	localListener.addr = l.Addr()
	defer l.Close()
	go localServer(l)

	if err := checkOpenSSLVersion(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v", err)
		os.Exit(1)
	}

	testConfig = &Config{
		Time:               func() time.Time { return time.Unix(0, 0) },
		Rand:               zeroSource{},
		Certificates:       make([]Certificate, 2),
		InsecureSkipVerify: true,
		CipherSuites:       allCipherSuites(),
		MinVersion:         VersionTLS10,
		MaxVersion:         VersionTLS13,
	}
	testConfig.Certificates[0].Certificate = [][]byte{testRSACertificate}
	testConfig.Certificates[0].PrivateKey = testRSAPrivateKey
	testConfig.Certificates[1].Certificate = [][]byte{testSNICertificate}
	testConfig.Certificates[1].PrivateKey = testRSAPrivateKey
	testConfig.BuildNameToCertificate()
	if *keyFile != "" {
		f, err := os.OpenFile(*keyFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			panic("failed to open -keylog file: " + err.Error())
		}
		testConfig.KeyLogWriter = f
		defer f.Close()
	}

	return m.Run()
}

func testHandshake(t *testing.T, clientConfig, serverConfig *Config) (serverState, clientState ConnectionState, err error) {
	const sentinel = "SENTINEL\n"
	c, s := localPipe(t)
	errChan := make(chan error)
	go func() {
		cli := Client(c, clientConfig)
		err := cli.Handshake()
		if err != nil {
			errChan <- fmt.Errorf("client: %v", err)
			c.Close()
			return
		}
		defer cli.Close()
		clientState = cli.ConnectionState()
		buf, err := io.ReadAll(cli)
		if err != nil {
			t.Errorf("failed to call cli.Read: %v", err)
		}
		if got := string(buf); got != sentinel {
			t.Errorf("read %q from TLS connection, but expected %q", got, sentinel)
		}
		errChan <- nil
	}()
	server := Server(s, serverConfig)
	err = server.Handshake()
	if err == nil {
		serverState = server.ConnectionState()
		if _, err := io.WriteString(server, sentinel); err != nil {
			t.Errorf("failed to call server.Write: %v", err)
		}
		if err := server.Close(); err != nil {
			t.Errorf("failed to call server.Close: %v", err)
		}
		err = <-errChan
	} else {
		s.Close()
		<-errChan
	}
	return
}

func fromHex(s string) []byte {
	b, _ := hex.DecodeString(s)
	return b
}

var testRSACertificate = fromHex("3082024b308201b4a003020102020900e8f09d3fe25beaa6300d06092a864886f70d01010b0500301f310b3009060355040a1302476f3110300e06035504031307476f20526f6f74301e170d3136303130313030303030305a170d3235303130313030303030305a301a310b3009060355040a1302476f310b300906035504031302476f30819f300d06092a864886f70d010101050003818d0030818902818100db467d932e12270648bc062821ab7ec4b6a25dfe1e5245887a3647a5080d92425bc281c0be97799840fb4f6d14fd2b138bc2a52e67d8d4099ed62238b74a0b74732bc234f1d193e596d9747bf3589f6c613cc0b041d4d92b2b2423775b1c3bbd755dce2054cfa163871d1e24c4f31d1a508baab61443ed97a77562f414c852d70203010001a38193308190300e0603551d0f0101ff0404030205a0301d0603551d250416301406082b0601050507030106082b06010505070302300c0603551d130101ff0402300030190603551d0e041204109f91161f43433e49a6de6db680d79f60301b0603551d230414301280104813494d137e1631bba301d5acab6e7b30190603551d1104123010820e6578616d706c652e676f6c616e67300d06092a864886f70d01010b0500038181009d30cc402b5b50a061cbbae55358e1ed8328a9581aa938a495a1ac315a1a84663d43d32dd90bf297dfd320643892243a00bccf9c7db74020015faad3166109a276fd13c3cce10c5ceeb18782f16c04ed73bbb343778d0c1cf10fa1d8408361c94c722b9daedb4606064df4c1b33ec0d1bd42d4dbfe3d1360845c21d33be9fae7")

var testRSACertificateIssuer = fromHex("3082021930820182a003020102020900ca5e4e811a965964300d06092a864886f70d01010b0500301f310b3009060355040a1302476f3110300e06035504031307476f20526f6f74301e170d3136303130313030303030305a170d3235303130313030303030305a301f310b3009060355040a1302476f3110300e06035504031307476f20526f6f7430819f300d06092a864886f70d010101050003818d0030818902818100d667b378bb22f34143b6cd2008236abefaf2852adf3ab05e01329e2c14834f5105df3f3073f99dab5442d45ee5f8f57b0111c8cb682fbb719a86944eebfffef3406206d898b8c1b1887797c9c5006547bb8f00e694b7a063f10839f269f2c34fff7a1f4b21fbcd6bfdfb13ac792d1d11f277b5c5b48600992203059f2a8f8cc50203010001a35d305b300e0603551d0f0101ff040403020204301d0603551d250416301406082b0601050507030106082b06010505070302300f0603551d130101ff040530030101ff30190603551d0e041204104813494d137e1631bba301d5acab6e7b300d06092a864886f70d01010b050003818100c1154b4bab5266221f293766ae4138899bd4c5e36b13cee670ceeaa4cbdf4f6679017e2fe649765af545749fe4249418a56bd38a04b81e261f5ce86b8d5c65413156a50d12449554748c59a30c515bc36a59d38bddf51173e899820b282e40aa78c806526fd184fb6b4cf186ec728edffa585440d2b3225325f7ab580e87dd76")

// testRSAPSSCertificate has signatureAlgorithm rsassaPss, but subjectPublicKeyInfo
// algorithm rsaEncryption, for use with the rsa_pss_rsae_* SignatureSchemes.
// See also TestRSAPSSKeyError. testRSAPSSCertificate is self-signed.
var testRSAPSSCertificate = fromHex("308202583082018da003020102021100f29926eb87ea8a0db9fcc247347c11b0304106092a864886f70d01010a3034a00f300d06096086480165030402010500a11c301a06092a864886f70d010108300d06096086480165030402010500a20302012030123110300e060355040a130741636d6520436f301e170d3137313132333136313631305a170d3138313132333136313631305a30123110300e060355040a130741636d6520436f30819f300d06092a864886f70d010101050003818d0030818902818100db467d932e12270648bc062821ab7ec4b6a25dfe1e5245887a3647a5080d92425bc281c0be97799840fb4f6d14fd2b138bc2a52e67d8d4099ed62238b74a0b74732bc234f1d193e596d9747bf3589f6c613cc0b041d4d92b2b2423775b1c3bbd755dce2054cfa163871d1e24c4f31d1a508baab61443ed97a77562f414c852d70203010001a3463044300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff04023000300f0603551d110408300687047f000001304106092a864886f70d01010a3034a00f300d06096086480165030402010500a11c301a06092a864886f70d010108300d06096086480165030402010500a20302012003818100cdac4ef2ce5f8d79881042707f7cbf1b5a8a00ef19154b40151771006cd41626e5496d56da0c1a139fd84695593cb67f87765e18aa03ea067522dd78d2a589b8c92364e12838ce346c6e067b51f1a7e6f4b37ffab13f1411896679d18e880e0ba09e302ac067efca460288e9538122692297ad8093d4f7dd701424d7700a46a1")

var testECDSACertificate = fromHex("3082020030820162020900b8bf2d47a0d2ebf4300906072a8648ce3d04013045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c7464301e170d3132313132323135303633325a170d3232313132303135303633325a3045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c746430819b301006072a8648ce3d020106052b81040023038186000400c4a1edbe98f90b4873367ec316561122f23d53c33b4d213dcd6b75e6f6b0dc9adf26c1bcb287f072327cb3642f1c90bcea6823107efee325c0483a69e0286dd33700ef0462dd0da09c706283d881d36431aa9e9731bd96b068c09b23de76643f1a5c7fe9120e5858b65f70dd9bd8ead5d7f5d5ccb9b69f30665b669a20e227e5bffe3b300906072a8648ce3d040103818c0030818802420188a24febe245c5487d1bacf5ed989dae4770c05e1bb62fbdf1b64db76140d311a2ceee0b7e927eff769dc33b7ea53fcefa10e259ec472d7cacda4e970e15a06fd00242014dfcbe67139c2d050ebd3fa38c25c13313830d9406bbd4377af6ec7ac9862eddd711697f857c56defb31782be4c7780daecbbe9e4e3624317b6a0f399512078f2a")

var testEd25519Certificate = fromHex("3082012e3081e1a00302010202100f431c425793941de987e4f1ad15005d300506032b657030123110300e060355040a130741636d6520436f301e170d3139303531363231333830315a170d3230303531353231333830315a30123110300e060355040a130741636d6520436f302a300506032b65700321003fe2152ee6e3ef3f4e854a7577a3649eede0bf842ccc92268ffa6f3483aaec8fa34d304b300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff0402300030160603551d11040f300d820b6578616d706c652e636f6d300506032b65700341006344ed9cc4be5324539fd2108d9fe82108909539e50dc155ff2c16b71dfcab7d4dd4e09313d0a942e0b66bfe5d6748d79f50bc6ccd4b03837cf20858cdaccf0c")

var testSNICertificate = fromHex("0441883421114c81480804c430820237308201a0a003020102020900e8f09d3fe25beaa6300d06092a864886f70d01010b0500301f310b3009060355040a1302476f3110300e06035504031307476f20526f6f74301e170d3136303130313030303030305a170d3235303130313030303030305a3023310b3009060355040a1302476f311430120603550403130b736e69746573742e636f6d30819f300d06092a864886f70d010101050003818d0030818902818100db467d932e12270648bc062821ab7ec4b6a25dfe1e5245887a3647a5080d92425bc281c0be97799840fb4f6d14fd2b138bc2a52e67d8d4099ed62238b74a0b74732bc234f1d193e596d9747bf3589f6c613cc0b041d4d92b2b2423775b1c3bbd755dce2054cfa163871d1e24c4f31d1a508baab61443ed97a77562f414c852d70203010001a3773075300e0603551d0f0101ff0404030205a0301d0603551d250416301406082b0601050507030106082b06010505070302300c0603551d130101ff0402300030190603551d0e041204109f91161f43433e49a6de6db680d79f60301b0603551d230414301280104813494d137e1631bba301d5acab6e7b300d06092a864886f70d01010b0500038181007beeecff0230dbb2e7a334af65430b7116e09f327c3bbf918107fc9c66cb497493207ae9b4dbb045cb63d605ec1b5dd485bb69124d68fa298dc776699b47632fd6d73cab57042acb26f083c4087459bc5a3bb3ca4d878d7fe31016b7bc9a627438666566e3389bfaeebe6becc9a0093ceed18d0f9ac79d56f3a73f18188988ed")

var testP256Certificate = fromHex("308201693082010ea00302010202105012dc24e1124ade4f3e153326ff27bf300a06082a8648ce3d04030230123110300e060355040a130741636d6520436f301e170d3137303533313232343934375a170d3138303533313232343934375a30123110300e060355040a130741636d6520436f3059301306072a8648ce3d020106082a8648ce3d03010703420004c02c61c9b16283bbcc14956d886d79b358aa614596975f78cece787146abf74c2d5dc578c0992b4f3c631373479ebf3892efe53d21c4f4f1cc9a11c3536b7f75a3463044300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff04023000300f0603551d1104083006820474657374300a06082a8648ce3d0403020349003046022100963712d6226c7b2bef41512d47e1434131aaca3ba585d666c924df71ac0448b3022100f4d05c725064741aef125f243cdbccaa2a5d485927831f221c43023bd5ae471a")

var testRSAPrivateKey, _ = x509.ParsePKCS1PrivateKey(fromHex("3082025b02010002818100db467d932e12270648bc062821ab7ec4b6a25dfe1e5245887a3647a5080d92425bc281c0be97799840fb4f6d14fd2b138bc2a52e67d8d4099ed62238b74a0b74732bc234f1d193e596d9747bf3589f6c613cc0b041d4d92b2b2423775b1c3bbd755dce2054cfa163871d1e24c4f31d1a508baab61443ed97a77562f414c852d702030100010281800b07fbcf48b50f1388db34b016298b8217f2092a7c9a04f77db6775a3d1279b62ee9951f7e371e9de33f015aea80660760b3951dc589a9f925ed7de13e8f520e1ccbc7498ce78e7fab6d59582c2386cc07ed688212a576ff37833bd5943483b5554d15a0b9b4010ed9bf09f207e7e9805f649240ed6c1256ed75ab7cd56d9671024100fded810da442775f5923debae4ac758390a032a16598d62f059bb2e781a9c2f41bfa015c209f966513fe3bf5a58717cbdb385100de914f88d649b7d15309fa49024100dd10978c623463a1802c52f012cfa72ff5d901f25a2292446552c2568b1840e49a312e127217c2186615aae4fb6602a4f6ebf3f3d160f3b3ad04c592f65ae41f02400c69062ca781841a09de41ed7a6d9f54adc5d693a2c6847949d9e1358555c9ac6a8d9e71653ac77beb2d3abaf7bb1183aa14278956575dbebf525d0482fd72d90240560fe1900ba36dae3022115fd952f2399fb28e2975a1c3e3d0b679660bdcb356cc189d611cfdd6d87cd5aea45aa30a2082e8b51e94c2f3dd5d5c6036a8a615ed0240143993d80ece56f877cb80048335701eb0e608cc0c1ca8c2227b52edf8f1ac99c562f2541b5ce81f0515af1c5b4770dba53383964b4b725ff46fdec3d08907df"))

var testECDSAPrivateKey, _ = x509.ParseECPrivateKey(fromHex("3081dc0201010442019883e909ad0ac9ea3d33f9eae661f1785206970f8ca9a91672f1eedca7a8ef12bd6561bb246dda5df4b4d5e7e3a92649bc5d83a0bf92972e00e62067d0c7bd99d7a00706052b81040023a18189038186000400c4a1edbe98f90b4873367ec316561122f23d53c33b4d213dcd6b75e6f6b0dc9adf26c1bcb287f072327cb3642f1c90bcea6823107efee325c0483a69e0286dd33700ef0462dd0da09c706283d881d36431aa9e9731bd96b068c09b23de76643f1a5c7fe9120e5858b65f70dd9bd8ead5d7f5d5ccb9b69f30665b669a20e227e5bffe3b"))

var testP256PrivateKey, _ = x509.ParseECPrivateKey(fromHex("30770201010420012f3b52bc54c36ba3577ad45034e2e8efe1e6999851284cb848725cfe029991a00a06082a8648ce3d030107a14403420004c02c61c9b16283bbcc14956d886d79b358aa614596975f78cece787146abf74c2d5dc578c0992b4f3c631373479ebf3892efe53d21c4f4f1cc9a11c3536b7f75"))

var testEd25519PrivateKey = ed25519.PrivateKey(fromHex("3a884965e76b3f55e5faf9615458a92354894234de3ec9f684d46d55cebf3dc63fe2152ee6e3ef3f4e854a7577a3649eede0bf842ccc92268ffa6f3483aaec8f"))

const clientCertificatePEM = `
-----BEGIN CERTIFICATE-----
MIIB7zCCAVigAwIBAgIQXBnBiWWDVW/cC8m5k5/pvDANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMB4XDTE2MDgxNzIxNTIzMVoXDTE3MDgxNzIxNTIz
MVowEjEQMA4GA1UEChMHQWNtZSBDbzCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkC
gYEAum+qhr3Pv5/y71yUYHhv6BPy0ZZvzdkybiI3zkH5yl0prOEn2mGi7oHLEMff
NFiVhuk9GeZcJ3NgyI14AvQdpJgJoxlwaTwlYmYqqyIjxXuFOE8uCXMyp70+m63K
hAfmDzr/d8WdQYUAirab7rCkPy1MTOZCPrtRyN1IVPQMjkcCAwEAAaNGMEQwDgYD
VR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMAwGA1UdEwEB/wQCMAAw
DwYDVR0RBAgwBocEfwAAATANBgkqhkiG9w0BAQsFAAOBgQBGq0Si+yhU+Fpn+GKU
8ZqyGJ7ysd4dfm92lam6512oFmyc9wnTN+RLKzZ8Aa1B0jLYw9KT+RBrjpW5LBeK
o0RIvFkTgxYEiKSBXCUNmAysEbEoVr4dzWFihAm/1oDGRY2CLLTYg5vbySK3KhIR
e/oCO8HJ/+rJnahJ05XX1Q7lNQ==
-----END CERTIFICATE-----`

var clientKeyPEM = testingKey(`
-----BEGIN RSA TESTING KEY-----
MIICXQIBAAKBgQC6b6qGvc+/n/LvXJRgeG/oE/LRlm/N2TJuIjfOQfnKXSms4Sfa
YaLugcsQx980WJWG6T0Z5lwnc2DIjXgC9B2kmAmjGXBpPCViZiqrIiPFe4U4Ty4J
czKnvT6brcqEB+YPOv93xZ1BhQCKtpvusKQ/LUxM5kI+u1HI3UhU9AyORwIDAQAB
AoGAEJZ03q4uuMb7b26WSQsOMeDsftdatT747LGgs3pNRkMJvTb/O7/qJjxoG+Mc
qeSj0TAZXp+PXXc3ikCECAc+R8rVMfWdmp903XgO/qYtmZGCorxAHEmR80SrfMXv
PJnznLQWc8U9nphQErR+tTESg7xWEzmFcPKwnZd1xg8ERYkCQQDTGtrFczlB2b/Z
9TjNMqUlMnTLIk/a/rPE2fLLmAYhK5sHnJdvDURaH2mF4nso0EGtENnTsh6LATnY
dkrxXGm9AkEA4hXHG2q3MnhgK1Z5hjv+Fnqd+8bcbII9WW4flFs15EKoMgS1w/PJ
zbsySaSy5IVS8XeShmT9+3lrleed4sy+UwJBAJOOAbxhfXP5r4+5R6ql66jES75w
jUCVJzJA5ORJrn8g64u2eGK28z/LFQbv9wXgCwfc72R468BdawFSLa/m2EECQGbZ
rWiFla26IVXV0xcD98VWJsTBZMlgPnSOqoMdM1kSEd4fUmlAYI/dFzV1XYSkOmVr
FhdZnklmpVDeu27P4c0CQQCuCOup0FlJSBpWY1TTfun/KMBkBatMz0VMA3d7FKIU
csPezl677Yjo8u1r/KzeI6zLg87Z8E6r6ZWNc9wBSZK6
-----END RSA TESTING KEY-----`)

const clientECDSACertificatePEM = `
-----BEGIN CERTIFICATE-----
MIIB/DCCAV4CCQCaMIRsJjXZFzAJBgcqhkjOPQQBMEUxCzAJBgNVBAYTAkFVMRMw
EQYDVQQIEwpTb21lLVN0YXRlMSEwHwYDVQQKExhJbnRlcm5ldCBXaWRnaXRzIFB0
eSBMdGQwHhcNMTIxMTE0MTMyNTUzWhcNMjIxMTEyMTMyNTUzWjBBMQswCQYDVQQG
EwJBVTEMMAoGA1UECBMDTlNXMRAwDgYDVQQHEwdQeXJtb250MRIwEAYDVQQDEwlK
b2VsIFNpbmcwgZswEAYHKoZIzj0CAQYFK4EEACMDgYYABACVjJF1FMBexFe01MNv
ja5oHt1vzobhfm6ySD6B5U7ixohLZNz1MLvT/2XMW/TdtWo+PtAd3kfDdq0Z9kUs
jLzYHQFMH3CQRnZIi4+DzEpcj0B22uCJ7B0rxE4wdihBsmKo+1vx+U56jb0JuK7q
ixgnTy5w/hOWusPTQBbNZU6sER7m8TAJBgcqhkjOPQQBA4GMADCBiAJCAOAUxGBg
C3JosDJdYUoCdFzCgbkWqD8pyDbHgf9stlvZcPE4O1BIKJTLCRpS8V3ujfK58PDa
2RU6+b0DeoeiIzXsAkIBo9SKeDUcSpoj0gq+KxAxnZxfvuiRs9oa9V2jI/Umi0Vw
jWVim34BmT0Y9hCaOGGbLlfk+syxis7iI6CH8OFnUes=
-----END CERTIFICATE-----`

var clientECDSAKeyPEM = testingKey(`
-----BEGIN EC PARAMETERS-----
BgUrgQQAIw==
-----END EC PARAMETERS-----
-----BEGIN EC TESTING KEY-----
MIHcAgEBBEIBkJN9X4IqZIguiEVKMqeBUP5xtRsEv4HJEtOpOGLELwO53SD78Ew8
k+wLWoqizS3NpQyMtrU8JFdWfj+C57UNkOugBwYFK4EEACOhgYkDgYYABACVjJF1
FMBexFe01MNvja5oHt1vzobhfm6ySD6B5U7ixohLZNz1MLvT/2XMW/TdtWo+PtAd
3kfDdq0Z9kUsjLzYHQFMH3CQRnZIi4+DzEpcj0B22uCJ7B0rxE4wdihBsmKo+1vx
+U56jb0JuK7qixgnTy5w/hOWusPTQBbNZU6sER7m8Q==
-----END EC TESTING KEY-----`)

const clientEd25519CertificatePEM = `
-----BEGIN CERTIFICATE-----
MIIBLjCB4aADAgECAhAX0YGTviqMISAQJRXoNCNPMAUGAytlcDASMRAwDgYDVQQK
EwdBY21lIENvMB4XDTE5MDUxNjIxNTQyNloXDTIwMDUxNTIxNTQyNlowEjEQMA4G
A1UEChMHQWNtZSBDbzAqMAUGAytlcAMhAAvgtWC14nkwPb7jHuBQsQTIbcd4bGkv
xRStmmNveRKRo00wSzAOBgNVHQ8BAf8EBAMCBaAwEwYDVR0lBAwwCgYIKwYBBQUH
AwIwDAYDVR0TAQH/BAIwADAWBgNVHREEDzANggtleGFtcGxlLmNvbTAFBgMrZXAD
QQD8GRcqlKUx+inILn9boF2KTjRAOdazENwZ/qAicbP1j6FYDc308YUkv+Y9FN/f
7Q7hF9gRomDQijcjKsJGqjoI
-----END CERTIFICATE-----`

var clientEd25519KeyPEM = testingKey(`
-----BEGIN TESTING KEY-----
MC4CAQAwBQYDK2VwBCIEINifzf07d9qx3d44e0FSbV4mC/xQxT644RRbpgNpin7I
-----END TESTING KEY-----`)
