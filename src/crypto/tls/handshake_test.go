// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bufio"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
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
// the test files to be regenerated. Generally one should combine the -update
// flag with -test.run to updated a specific test. Since the reference
// implementation will always generate fresh random numbers, large parts of
// the reference connection will always change.

var (
	update = flag.Bool("update", false, "update golden files on disk")

	opensslVersionTestOnce sync.Once
	opensslVersionTestErr  error
)

func checkOpenSSLVersion(t *testing.T) {
	opensslVersionTestOnce.Do(testOpenSSLVersion)
	if opensslVersionTestErr != nil {
		t.Fatal(opensslVersionTestErr)
	}
}

func testOpenSSLVersion() {
	// This test ensures that the version of OpenSSL looks reasonable
	// before updating the test data.

	if !*update {
		return
	}

	openssl := exec.Command("openssl", "version")
	output, err := openssl.CombinedOutput()
	if err != nil {
		opensslVersionTestErr = err
		return
	}

	version := string(output)
	if strings.HasPrefix(version, "OpenSSL 1.1.0") {
		return
	}

	println("***********************************************")
	println("")
	println("You need to build OpenSSL 1.1.0 from source in order")
	println("to update the test data.")
	println("")
	println("Configure it with:")
	println("./Configure enable-weak-ssl-ciphers enable-ssl3 enable-ssl3-method -static linux-x86_64")
	println("and then add the apps/ directory at the front of your PATH.")
	println("***********************************************")

	opensslVersionTestErr = errors.New("version of OpenSSL does not appear to be suitable for updating test data")
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

		if i := strings.IndexByte(line, ' '); i >= 0 {
			line = line[i:]
		} else {
			return nil, errors.New("invalid test data")
		}

		if i := strings.IndexByte(line, '|'); i >= 0 {
			line = line[:i]
		} else {
			return nil, errors.New("invalid test data")
		}

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
	file, err := ioutil.TempFile("", "go-tls-test")
	if err != nil {
		panic("failed to create temp file: " + err.Error())
	}
	path := file.Name()
	file.WriteString(contents)
	file.Close()
	return path
}
