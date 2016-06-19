// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"testing"
)

const (
	twain       = "testdata/Mark.Twain-Tom.Sawyer.txt"
	twainLen    = 387851
	twainSHA256 = "461eb7cb2d57d293fc680c836464c9125e4382be3596f7d415093ae9db8fcb0e"
)

func TestSendfile(t *testing.T) {
	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	errc := make(chan error, 1)
	go func(ln Listener) {
		// Wait for a connection.
		conn, err := ln.Accept()
		if err != nil {
			errc <- err
			close(errc)
			return
		}

		go func() {
			defer close(errc)
			defer conn.Close()

			f, err := os.Open(twain)
			if err != nil {
				errc <- err
				return
			}
			defer f.Close()

			// Return file data using io.Copy, which should use
			// sendFile if available.
			sbytes, err := io.Copy(conn, f)
			if err != nil {
				errc <- err
				return
			}

			if sbytes != twainLen {
				errc <- fmt.Errorf("sent %d bytes; expected %d", sbytes, twainLen)
				return
			}
		}()
	}(ln)

	// Connect to listener to retrieve file and verify digest matches
	// expected.
	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	h := sha256.New()
	rbytes, err := io.Copy(h, c)
	if err != nil {
		t.Error(err)
	}

	if rbytes != twainLen {
		t.Errorf("received %d bytes; expected %d", rbytes, twainLen)
	}

	if res := hex.EncodeToString(h.Sum(nil)); res != twainSHA256 {
		t.Error("retrieved data hash did not match")
	}

	for err := range errc {
		t.Error(err)
	}
}
