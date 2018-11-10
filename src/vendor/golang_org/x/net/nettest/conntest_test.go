// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.8

package nettest

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"runtime"
	"testing"
)

// testUnixAddr uses ioutil.TempFile to get a name that is unique.
// It also uses /tmp directory in case it is prohibited to create UNIX
// sockets in TMPDIR.
func testUnixAddr() string {
	f, err := ioutil.TempFile("", "go-nettest")
	if err != nil {
		panic(err)
	}
	addr := f.Name()
	f.Close()
	os.Remove(addr)
	return addr
}

// testableNetwork reports whether network is testable on the current
// platform configuration.
// This is based on logic from standard library's net/platform_test.go.
func testableNetwork(network string) bool {
	switch network {
	case "unix":
		switch runtime.GOOS {
		case "android", "nacl", "plan9", "windows":
			return false
		}
		if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
			return false
		}
	case "unixpacket":
		switch runtime.GOOS {
		case "android", "darwin", "nacl", "plan9", "windows", "freebsd":
			return false
		}
	}
	return true
}

func newLocalListener(network string) (net.Listener, error) {
	switch network {
	case "tcp":
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			ln, err = net.Listen("tcp6", "[::1]:0")
		}
		return ln, err
	case "unix", "unixpacket":
		return net.Listen(network, testUnixAddr())
	}
	return nil, fmt.Errorf("%s is not supported", network)
}

func TestTestConn(t *testing.T) {
	tests := []struct{ name, network string }{
		{"TCP", "tcp"},
		{"UnixPipe", "unix"},
		{"UnixPacketPipe", "unixpacket"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !testableNetwork(tt.network) {
				t.Skipf("not supported on %s", runtime.GOOS)
			}

			mp := func() (c1, c2 net.Conn, stop func(), err error) {
				ln, err := newLocalListener(tt.network)
				if err != nil {
					return nil, nil, nil, err
				}

				// Start a connection between two endpoints.
				var err1, err2 error
				done := make(chan bool)
				go func() {
					c2, err2 = ln.Accept()
					close(done)
				}()
				c1, err1 = net.Dial(ln.Addr().Network(), ln.Addr().String())
				<-done

				stop = func() {
					if err1 == nil {
						c1.Close()
					}
					if err2 == nil {
						c2.Close()
					}
					ln.Close()
					switch tt.network {
					case "unix", "unixpacket":
						os.Remove(ln.Addr().String())
					}
				}

				switch {
				case err1 != nil:
					stop()
					return nil, nil, nil, err1
				case err2 != nil:
					stop()
					return nil, nil, nil, err2
				default:
					return c1, c2, stop, nil
				}
			}

			TestConn(t, mp)
		})
	}
}
