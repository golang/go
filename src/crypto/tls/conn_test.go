// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"io"
	"net"
	"testing"
)

func TestRoundUp(t *testing.T) {
	if roundUp(0, 16) != 0 ||
		roundUp(1, 16) != 16 ||
		roundUp(15, 16) != 16 ||
		roundUp(16, 16) != 16 ||
		roundUp(17, 16) != 32 {
		t.Error("roundUp broken")
	}
}

// will be initialized with {0, 255, 255, ..., 255}
var padding255Bad = [256]byte{}

// will be initialized with {255, 255, 255, ..., 255}
var padding255Good = [256]byte{255}

var paddingTests = []struct {
	in          []byte
	good        bool
	expectedLen int
}{
	{[]byte{1, 2, 3, 4, 0}, true, 4},
	{[]byte{1, 2, 3, 4, 0, 1}, false, 0},
	{[]byte{1, 2, 3, 4, 99, 99}, false, 0},
	{[]byte{1, 2, 3, 4, 1, 1}, true, 4},
	{[]byte{1, 2, 3, 2, 2, 2}, true, 3},
	{[]byte{1, 2, 3, 3, 3, 3}, true, 2},
	{[]byte{1, 2, 3, 4, 3, 3}, false, 0},
	{[]byte{1, 4, 4, 4, 4, 4}, true, 1},
	{[]byte{5, 5, 5, 5, 5, 5}, true, 0},
	{[]byte{6, 6, 6, 6, 6, 6}, false, 0},
	{padding255Bad[:], false, 0},
	{padding255Good[:], true, 0},
}

func TestRemovePadding(t *testing.T) {
	for i := 1; i < len(padding255Bad); i++ {
		padding255Bad[i] = 255
		padding255Good[i] = 255
	}
	for i, test := range paddingTests {
		paddingLen, good := extractPadding(test.in)
		expectedGood := byte(255)
		if !test.good {
			expectedGood = 0
		}
		if good != expectedGood {
			t.Errorf("#%d: wrong validity, want:%d got:%d", i, expectedGood, good)
		}
		if good == 255 && len(test.in)-paddingLen != test.expectedLen {
			t.Errorf("#%d: got %d, want %d", i, len(test.in)-paddingLen, test.expectedLen)
		}
	}
}

var certExampleCom = `308201713082011ba003020102021005a75ddf21014d5f417083b7a010ba2e300d06092a864886f70d01010b050030123110300e060355040a130741636d6520436f301e170d3136303831373231343135335a170d3137303831373231343135335a30123110300e060355040a130741636d6520436f305c300d06092a864886f70d0101010500034b003048024100b37f0fdd67e715bf532046ac34acbd8fdc4dabe2b598588f3f58b1f12e6219a16cbfe54d2b4b665396013589262360b6721efa27d546854f17cc9aeec6751db10203010001a34d304b300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff0402300030160603551d11040f300d820b6578616d706c652e636f6d300d06092a864886f70d01010b050003410059fc487866d3d855503c8e064ca32aac5e9babcece89ec597f8b2b24c17867f4a5d3b4ece06e795bfc5448ccbd2ffca1b3433171ebf3557a4737b020565350a0`

var certWildcardExampleCom = `308201743082011ea003020102021100a7aa6297c9416a4633af8bec2958c607300d06092a864886f70d01010b050030123110300e060355040a130741636d6520436f301e170d3136303831373231343231395a170d3137303831373231343231395a30123110300e060355040a130741636d6520436f305c300d06092a864886f70d0101010500034b003048024100b105afc859a711ee864114e7d2d46c2dcbe392d3506249f6c2285b0eb342cc4bf2d803677c61c0abde443f084745c1a6d62080e5664ef2cc8f50ad8a0ab8870b0203010001a34f304d300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff0402300030180603551d110411300f820d2a2e6578616d706c652e636f6d300d06092a864886f70d01010b0500034100af26088584d266e3f6566360cf862c7fecc441484b098b107439543144a2b93f20781988281e108c6d7656934e56950e1e5f2bcf38796b814ccb729445856c34`

var certFooExampleCom = `308201753082011fa00302010202101bbdb6070b0aeffc49008cde74deef29300d06092a864886f70d01010b050030123110300e060355040a130741636d6520436f301e170d3136303831373231343234345a170d3137303831373231343234345a30123110300e060355040a130741636d6520436f305c300d06092a864886f70d0101010500034b003048024100f00ac69d8ca2829f26216c7b50f1d4bbabad58d447706476cd89a2f3e1859943748aa42c15eedc93ac7c49e40d3b05ed645cb6b81c4efba60d961f44211a54eb0203010001a351304f300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff04023000301a0603551d1104133011820f666f6f2e6578616d706c652e636f6d300d06092a864886f70d01010b0500034100a0957fca6d1e0f1ef4b247348c7a8ca092c29c9c0ecc1898ea6b8065d23af6d922a410dd2335a0ea15edd1394cef9f62c9e876a21e35250a0b4fe1ddceba0f36`

var certDoubleWildcardExampleCom = `308201753082011fa003020102021039d262d8538db8ffba30d204e02ddeb5300d06092a864886f70d01010b050030123110300e060355040a130741636d6520436f301e170d3136303831373231343331335a170d3137303831373231343331335a30123110300e060355040a130741636d6520436f305c300d06092a864886f70d0101010500034b003048024100abb6bd84b8b9be3fb9415d00f22b4ddcaec7c99855b9d818c09003e084578430e5cfd2e35faa3561f036d496aa43a9ca6e6cf23c72a763c04ae324004f6cbdbb0203010001a351304f300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff04023000301a0603551d1104133011820f2a2e2a2e6578616d706c652e636f6d300d06092a864886f70d01010b05000341004837521004a5b6bc7ad5d6c0dae60bb7ee0fa5e4825be35e2bb6ef07ee29396ca30ceb289431bcfd363888ba2207139933ac7c6369fa8810c819b2e2966abb4b`

func TestCertificateSelection(t *testing.T) {
	config := Config{
		Certificates: []Certificate{
			{
				Certificate: [][]byte{fromHex(certExampleCom)},
			},
			{
				Certificate: [][]byte{fromHex(certWildcardExampleCom)},
			},
			{
				Certificate: [][]byte{fromHex(certFooExampleCom)},
			},
			{
				Certificate: [][]byte{fromHex(certDoubleWildcardExampleCom)},
			},
		},
	}

	config.BuildNameToCertificate()

	pointerToIndex := func(c *Certificate) int {
		for i := range config.Certificates {
			if c == &config.Certificates[i] {
				return i
			}
		}
		return -1
	}

	certificateForName := func(name string) *Certificate {
		clientHello := &ClientHelloInfo{
			ServerName: name,
		}
		if cert, err := config.getCertificate(clientHello); err != nil {
			t.Errorf("unable to get certificate for name '%s': %s", name, err)
			return nil
		} else {
			return cert
		}
	}

	if n := pointerToIndex(certificateForName("example.com")); n != 0 {
		t.Errorf("example.com returned certificate %d, not 0", n)
	}
	if n := pointerToIndex(certificateForName("bar.example.com")); n != 1 {
		t.Errorf("bar.example.com returned certificate %d, not 1", n)
	}
	if n := pointerToIndex(certificateForName("foo.example.com")); n != 2 {
		t.Errorf("foo.example.com returned certificate %d, not 2", n)
	}
	if n := pointerToIndex(certificateForName("foo.bar.example.com")); n != 3 {
		t.Errorf("foo.bar.example.com returned certificate %d, not 3", n)
	}
	if n := pointerToIndex(certificateForName("foo.bar.baz.example.com")); n != 0 {
		t.Errorf("foo.bar.baz.example.com returned certificate %d, not 0", n)
	}
}

// Run with multiple crypto configs to test the logic for computing TLS record overheads.
func runDynamicRecordSizingTest(t *testing.T, config *Config) {
	clientConn, serverConn := localPipe(t)

	serverConfig := config.Clone()
	serverConfig.DynamicRecordSizingDisabled = false
	tlsConn := Server(serverConn, serverConfig)

	handshakeDone := make(chan struct{})
	recordSizesChan := make(chan []int, 1)
	defer func() { <-recordSizesChan }() // wait for the goroutine to exit
	go func() {
		// This goroutine performs a TLS handshake over clientConn and
		// then reads TLS records until EOF. It writes a slice that
		// contains all the record sizes to recordSizesChan.
		defer close(recordSizesChan)
		defer clientConn.Close()

		tlsConn := Client(clientConn, config)
		if err := tlsConn.Handshake(); err != nil {
			t.Errorf("Error from client handshake: %v", err)
			return
		}
		close(handshakeDone)

		var recordHeader [recordHeaderLen]byte
		var record []byte
		var recordSizes []int

		for {
			n, err := io.ReadFull(clientConn, recordHeader[:])
			if err == io.EOF {
				break
			}
			if err != nil || n != len(recordHeader) {
				t.Errorf("io.ReadFull = %d, %v", n, err)
				return
			}

			length := int(recordHeader[3])<<8 | int(recordHeader[4])
			if len(record) < length {
				record = make([]byte, length)
			}

			n, err = io.ReadFull(clientConn, record[:length])
			if err != nil || n != length {
				t.Errorf("io.ReadFull = %d, %v", n, err)
				return
			}

			recordSizes = append(recordSizes, recordHeaderLen+length)
		}

		recordSizesChan <- recordSizes
	}()

	if err := tlsConn.Handshake(); err != nil {
		t.Fatalf("Error from server handshake: %s", err)
	}
	<-handshakeDone

	// The server writes these plaintexts in order.
	plaintext := bytes.Join([][]byte{
		bytes.Repeat([]byte("x"), recordSizeBoostThreshold),
		bytes.Repeat([]byte("y"), maxPlaintext*2),
		bytes.Repeat([]byte("z"), maxPlaintext),
	}, nil)

	if _, err := tlsConn.Write(plaintext); err != nil {
		t.Fatalf("Error from server write: %s", err)
	}
	if err := tlsConn.Close(); err != nil {
		t.Fatalf("Error from server close: %s", err)
	}

	recordSizes := <-recordSizesChan
	if recordSizes == nil {
		t.Fatalf("Client encountered an error")
	}

	// Drop the size of the second to last record, which is likely to be
	// truncated, and the last record, which is a close_notify alert.
	recordSizes = recordSizes[:len(recordSizes)-2]

	// recordSizes should contain a series of records smaller than
	// tcpMSSEstimate followed by some larger than maxPlaintext.
	seenLargeRecord := false
	for i, size := range recordSizes {
		if !seenLargeRecord {
			if size > (i+1)*tcpMSSEstimate {
				t.Fatalf("Record #%d has size %d, which is too large too soon", i, size)
			}
			if size >= maxPlaintext {
				seenLargeRecord = true
			}
		} else if size <= maxPlaintext {
			t.Fatalf("Record #%d has size %d but should be full sized", i, size)
		}
	}

	if !seenLargeRecord {
		t.Fatalf("No large records observed")
	}
}

func TestDynamicRecordSizingWithStreamCipher(t *testing.T) {
	config := testConfig.Clone()
	config.MaxVersion = VersionTLS12
	config.CipherSuites = []uint16{TLS_RSA_WITH_RC4_128_SHA}
	runDynamicRecordSizingTest(t, config)
}

func TestDynamicRecordSizingWithCBC(t *testing.T) {
	config := testConfig.Clone()
	config.MaxVersion = VersionTLS12
	config.CipherSuites = []uint16{TLS_RSA_WITH_AES_256_CBC_SHA}
	runDynamicRecordSizingTest(t, config)
}

func TestDynamicRecordSizingWithAEAD(t *testing.T) {
	config := testConfig.Clone()
	config.MaxVersion = VersionTLS12
	config.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}
	runDynamicRecordSizingTest(t, config)
}

func TestDynamicRecordSizingWithTLSv13(t *testing.T) {
	config := testConfig.Clone()
	runDynamicRecordSizingTest(t, config)
}

// hairpinConn is a net.Conn that makes a “hairpin” call when closed, back into
// the tls.Conn which is calling it.
type hairpinConn struct {
	net.Conn
	tlsConn *Conn
}

func (conn *hairpinConn) Close() error {
	conn.tlsConn.ConnectionState()
	return nil
}

func TestHairpinInClose(t *testing.T) {
	// This tests that the underlying net.Conn can call back into the
	// tls.Conn when being closed without deadlocking.
	client, server := localPipe(t)
	defer server.Close()
	defer client.Close()

	conn := &hairpinConn{client, nil}
	tlsConn := Server(conn, &Config{
		GetCertificate: func(*ClientHelloInfo) (*Certificate, error) {
			panic("unreachable")
		},
	})
	conn.tlsConn = tlsConn

	// This call should not deadlock.
	tlsConn.Close()
}
