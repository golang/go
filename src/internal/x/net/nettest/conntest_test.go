// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.8

package nettest

import (
	"net"
	"os"
	"runtime"
	"testing"

	"internal/x/net/internal/nettest"
)

func TestTestConn(t *testing.T) {
	tests := []struct{ name, network string }{
		{"TCP", "tcp"},
		{"UnixPipe", "unix"},
		{"UnixPacketPipe", "unixpacket"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !nettest.TestableNetwork(tt.network) {
				t.Skipf("not supported on %s", runtime.GOOS)
			}

			mp := func() (c1, c2 net.Conn, stop func(), err error) {
				ln, err := nettest.NewLocalListener(tt.network)
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
