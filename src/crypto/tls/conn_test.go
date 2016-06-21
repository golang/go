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
}

func TestRemovePadding(t *testing.T) {
	for i, test := range paddingTests {
		payload, good := removePadding(test.in)
		expectedGood := byte(255)
		if !test.good {
			expectedGood = 0
		}
		if good != expectedGood {
			t.Errorf("#%d: wrong validity, want:%d got:%d", i, expectedGood, good)
		}
		if good == 255 && len(payload) != test.expectedLen {
			t.Errorf("#%d: got %d, want %d", i, len(payload), test.expectedLen)
		}
	}
}

var certExampleCom = `308201403081eda003020102020101300b06092a864886f70d010105301e311c301a060355040a131354657374696e67204365727469666963617465301e170d3131313030313138353835325a170d3132303933303138353835325a301e311c301a060355040a131354657374696e67204365727469666963617465305a300b06092a864886f70d010101034b003048024100bced6e32368599eeddf18796bfd03958a154f87e5b084f96e85136a56b886733592f493f0fc68b0d6b3551781cb95e13c5de458b28d6fb60d20a9129313261410203010001a31a301830160603551d11040f300d820b6578616d706c652e636f6d300b06092a864886f70d0101050341001a0b419d2c74474c6450654e5f10b32bf426ffdf55cad1c52602e7a9151513a3424c70f5960dcd682db0c33769cc1daa3fcdd3db10809d2392ed4a1bf50ced18`

var certWildcardExampleCom = `308201423081efa003020102020101300b06092a864886f70d010105301e311c301a060355040a131354657374696e67204365727469666963617465301e170d3131313030313139303034365a170d3132303933303139303034365a301e311c301a060355040a131354657374696e67204365727469666963617465305a300b06092a864886f70d010101034b003048024100bced6e32368599eeddf18796bfd03958a154f87e5b084f96e85136a56b886733592f493f0fc68b0d6b3551781cb95e13c5de458b28d6fb60d20a9129313261410203010001a31c301a30180603551d110411300f820d2a2e6578616d706c652e636f6d300b06092a864886f70d0101050341001676f0c9e7c33c1b656ed5a6476c4e2ee9ec8e62df7407accb1875272b2edd0a22096cb2c22598d11604104d604f810eb4b5987ca6bb319c7e6ce48725c54059`

var certFooExampleCom = `308201443081f1a003020102020101300b06092a864886f70d010105301e311c301a060355040a131354657374696e67204365727469666963617465301e170d3131313030313139303131345a170d3132303933303139303131345a301e311c301a060355040a131354657374696e67204365727469666963617465305a300b06092a864886f70d010101034b003048024100bced6e32368599eeddf18796bfd03958a154f87e5b084f96e85136a56b886733592f493f0fc68b0d6b3551781cb95e13c5de458b28d6fb60d20a9129313261410203010001a31e301c301a0603551d1104133011820f666f6f2e6578616d706c652e636f6d300b06092a864886f70d010105034100646a2a51f2aa2477add854b462cf5207ba16d3213ffb5d3d0eed473fbf09935019192d1d5b8ca6a2407b424cf04d97c4cd9197c83ecf81f0eab9464a1109d09f`

var certDoubleWildcardExampleCom = `308201443081f1a003020102020101300b06092a864886f70d010105301e311c301a060355040a131354657374696e67204365727469666963617465301e170d3131313030313139303134315a170d3132303933303139303134315a301e311c301a060355040a131354657374696e67204365727469666963617465305a300b06092a864886f70d010101034b003048024100bced6e32368599eeddf18796bfd03958a154f87e5b084f96e85136a56b886733592f493f0fc68b0d6b3551781cb95e13c5de458b28d6fb60d20a9129313261410203010001a31e301c301a0603551d1104133011820f2a2e2a2e6578616d706c652e636f6d300b06092a864886f70d0101050341001c3de267975f56ef57771c6218ef95ecc65102e57bd1defe6f7efea90d9b26cf40de5bd7ad75e46201c7f2a92aaa3e907451e9409f65e28ddb6db80d726290f6`

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
	clientConn, serverConn := net.Pipe()

	serverConfig := config.clone()
	serverConfig.DynamicRecordSizingDisabled = false
	tlsConn := Server(serverConn, serverConfig)

	recordSizesChan := make(chan []int, 1)
	go func() {
		// This goroutine performs a TLS handshake over clientConn and
		// then reads TLS records until EOF. It writes a slice that
		// contains all the record sizes to recordSizesChan.
		defer close(recordSizesChan)
		defer clientConn.Close()

		tlsConn := Client(clientConn, config)
		if err := tlsConn.Handshake(); err != nil {
			t.Errorf("Error from client handshake: %s", err)
			return
		}

		var recordHeader [recordHeaderLen]byte
		var record []byte
		var recordSizes []int

		for {
			n, err := clientConn.Read(recordHeader[:])
			if err == io.EOF {
				break
			}
			if err != nil || n != len(recordHeader) {
				t.Errorf("Error from client read: %s", err)
				return
			}

			length := int(recordHeader[3])<<8 | int(recordHeader[4])
			if len(record) < length {
				record = make([]byte, length)
			}

			n, err = clientConn.Read(record[:length])
			if err != nil || n != length {
				t.Errorf("Error from client read: %s", err)
				return
			}

			// The last record will be a close_notify alert, which
			// we don't wish to record.
			if recordType(recordHeader[0]) == recordTypeApplicationData {
				recordSizes = append(recordSizes, recordHeaderLen+length)
			}
		}

		recordSizesChan <- recordSizes
	}()

	if err := tlsConn.Handshake(); err != nil {
		t.Fatalf("Error from server handshake: %s", err)
	}

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

	// Drop the size of last record, which is likely to be truncated.
	recordSizes = recordSizes[:len(recordSizes)-1]

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
	config := testConfig.clone()
	config.CipherSuites = []uint16{TLS_RSA_WITH_RC4_128_SHA}
	runDynamicRecordSizingTest(t, config)
}

func TestDynamicRecordSizingWithCBC(t *testing.T) {
	config := testConfig.clone()
	config.CipherSuites = []uint16{TLS_RSA_WITH_AES_256_CBC_SHA}
	runDynamicRecordSizingTest(t, config)
}

func TestDynamicRecordSizingWithAEAD(t *testing.T) {
	config := testConfig.clone()
	config.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}
	runDynamicRecordSizingTest(t, config)
}
