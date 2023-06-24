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

	// We're unable to pass port 0 here (let the OS choose a spare port).
	// Although wasmtime accepts port 0, and testdata/main.go successfully
	// listens, there's no way for this test case to query the chosen port
	// so that it can connect to the WASM module. The WASM module itself
	// cannot access any information about the socket due to limitations
	// with WASI preview 1 networking, and wasmtime does not log the address
	// when you preopen a socket. Instead, we probe for a free port here.
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
	case "wasmtime":
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
	var err error
	for attempts := 0; attempts < 5; attempts++ {
		conn, err = net.Dial("tcp", host)
		if err == nil {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	if err != nil {
		t.Log(b.String())
		t.Fatal(err)
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
