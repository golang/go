// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasi_test

import (
	"bytes"
	"fmt"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestTCPEcho(t *testing.T) {
	if target != "wasip1/wasm" {
		t.Skip()
	}

	// We're unable to use port 0 here (let the OS choose a spare port).
	// Although the WASM runtime accepts port 0, and the WASM module listens
	// successfully, there's no way for this test to query the selected port
	// so that it can connect to the WASM module. The WASM module itself
	// cannot access any information about the socket due to limitations
	// with WASI preview 1 networking, and the WASM runtimes do not log the
	// port when you pre-open a socket. So, we probe for a free port here.
	// Given there's an unavoidable race condition, the test is disabled by
	// default.
	if os.Getenv("GOWASIENABLERACYTEST") != "1" {
		t.Skip("skipping WASI test with unavoidable race condition")
	}
	var host string
	port := rand.Intn(10000) + 40000
	for attempts := 0; attempts < 10; attempts++ {
		host = fmt.Sprintf("127.0.0.1:%d", port)
		l, err := net.Listen("tcp", host)
		if err == nil {
			l.Close()
			break
		}
		port++
	}

	subProcess := exec.Command("go", "run", "./testdata/tcpecho.go")

	subProcess.Env = append(os.Environ(), "GOOS=wasip1", "GOARCH=wasm")

	switch os.Getenv("GOWASIRUNTIME") {
	case "wazero":
		subProcess.Env = append(subProcess.Env, "GOWASIRUNTIMEARGS=--listen="+host)
	case "wasmtime", "":
		subProcess.Env = append(subProcess.Env, "GOWASIRUNTIMEARGS=--tcplisten="+host)
	default:
		t.Skip("WASI runtime does not support sockets")
	}

	var b bytes.Buffer
	subProcess.Stdout = &b
	subProcess.Stderr = &b

	if err := subProcess.Start(); err != nil {
		t.Log(b.String())
		t.Fatal(err)
	}
	defer subProcess.Process.Kill()

	var conn net.Conn
	for {
		var err error
		conn, err = net.Dial("tcp", host)
		if err == nil {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	defer conn.Close()

	payload := []byte("foobar")
	if _, err := conn.Write(payload); err != nil {
		t.Fatal(err)
	}
	var buf [256]byte
	n, err := conn.Read(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if string(buf[:n]) != string(payload) {
		t.Error("unexpected payload")
		t.Logf("expect: %d bytes (%v)", len(payload), payload)
		t.Logf("actual: %d bytes (%v)", n, buf[:n])
	}
}
