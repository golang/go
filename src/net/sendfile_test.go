// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !js

package net

import (
	"bytes"
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

func TestSendfileParts(t *testing.T) {
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

			for i := 0; i < 3; i++ {
				// Return file data using io.CopyN, which should use
				// sendFile if available.
				_, err = io.CopyN(conn, f, 3)
				if err != nil {
					errc <- err
					return
				}
			}
		}()
	}(ln)

	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	buf := new(bytes.Buffer)
	buf.ReadFrom(c)

	if want, have := "Produced ", buf.String(); have != want {
		t.Errorf("unexpected server reply %q, want %q", have, want)
	}

	for err := range errc {
		t.Error(err)
	}
}

func TestSendfileSeeked(t *testing.T) {
	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	const seekTo = 65 << 10
	const sendSize = 10 << 10

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
			if _, err := f.Seek(seekTo, os.SEEK_SET); err != nil {
				errc <- err
				return
			}

			_, err = io.CopyN(conn, f, sendSize)
			if err != nil {
				errc <- err
				return
			}
		}()
	}(ln)

	c, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	buf := new(bytes.Buffer)
	buf.ReadFrom(c)

	if buf.Len() != sendSize {
		t.Errorf("Got %d bytes; want %d", buf.Len(), sendSize)
	}

	for err := range errc {
		t.Error(err)
	}
}
