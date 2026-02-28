// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"os"
	"testing"
)

var tcpServerTests = []struct {
	snet, saddr string // server endpoint
	tnet, taddr string // target endpoint for client
}{
	{snet: "tcp", saddr: ":0", tnet: "tcp", taddr: "127.0.0.1"},
	{snet: "tcp", saddr: "0.0.0.0:0", tnet: "tcp", taddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]:0", tnet: "tcp", taddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::]:0", tnet: "tcp", taddr: "::1"},

	{snet: "tcp", saddr: ":0", tnet: "tcp", taddr: "::1"},
	{snet: "tcp", saddr: "0.0.0.0:0", tnet: "tcp", taddr: "::1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]:0", tnet: "tcp", taddr: "::1"},
	{snet: "tcp", saddr: "[::]:0", tnet: "tcp", taddr: "127.0.0.1"},

	{snet: "tcp", saddr: ":0", tnet: "tcp4", taddr: "127.0.0.1"},
	{snet: "tcp", saddr: "0.0.0.0:0", tnet: "tcp4", taddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]:0", tnet: "tcp4", taddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::]:0", tnet: "tcp6", taddr: "::1"},

	{snet: "tcp", saddr: ":0", tnet: "tcp6", taddr: "::1"},
	{snet: "tcp", saddr: "0.0.0.0:0", tnet: "tcp6", taddr: "::1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]:0", tnet: "tcp6", taddr: "::1"},
	{snet: "tcp", saddr: "[::]:0", tnet: "tcp4", taddr: "127.0.0.1"},

	{snet: "tcp", saddr: "127.0.0.1:0", tnet: "tcp", taddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:127.0.0.1]:0", tnet: "tcp", taddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::1]:0", tnet: "tcp", taddr: "::1"},

	{snet: "tcp4", saddr: ":0", tnet: "tcp4", taddr: "127.0.0.1"},
	{snet: "tcp4", saddr: "0.0.0.0:0", tnet: "tcp4", taddr: "127.0.0.1"},
	{snet: "tcp4", saddr: "[::ffff:0.0.0.0]:0", tnet: "tcp4", taddr: "127.0.0.1"},

	{snet: "tcp4", saddr: "127.0.0.1:0", tnet: "tcp4", taddr: "127.0.0.1"},

	{snet: "tcp6", saddr: ":0", tnet: "tcp6", taddr: "::1"},
	{snet: "tcp6", saddr: "[::]:0", tnet: "tcp6", taddr: "::1"},

	{snet: "tcp6", saddr: "[::1]:0", tnet: "tcp6", taddr: "::1"},
}

// TestTCPServer tests concurrent accept-read-write servers.
func TestTCPServer(t *testing.T) {
	const N = 3

	for i, tt := range tcpServerTests {
		t.Run(tt.snet+" "+tt.saddr+"<-"+tt.taddr, func(t *testing.T) {
			if !testableListenArgs(tt.snet, tt.saddr, tt.taddr) {
				t.Skip("not testable")
			}

			ln, err := Listen(tt.snet, tt.saddr)
			if err != nil {
				if perr := parseDialError(err); perr != nil {
					t.Error(perr)
				}
				t.Fatal(err)
			}

			var lss []*localServer
			var tpchs []chan error
			defer func() {
				for _, ls := range lss {
					ls.teardown()
				}
			}()
			for i := 0; i < N; i++ {
				ls := (&streamListener{Listener: ln}).newLocalServer()
				lss = append(lss, ls)
				tpchs = append(tpchs, make(chan error, 1))
			}
			for i := 0; i < N; i++ {
				ch := tpchs[i]
				handler := func(ls *localServer, ln Listener) { ls.transponder(ln, ch) }
				if err := lss[i].buildup(handler); err != nil {
					t.Fatal(err)
				}
			}

			var trchs []chan error
			for i := 0; i < N; i++ {
				_, port, err := SplitHostPort(lss[i].Listener.Addr().String())
				if err != nil {
					t.Fatal(err)
				}
				d := Dialer{Timeout: someTimeout}
				c, err := d.Dial(tt.tnet, JoinHostPort(tt.taddr, port))
				if err != nil {
					if perr := parseDialError(err); perr != nil {
						t.Error(perr)
					}
					t.Fatal(err)
				}
				defer c.Close()
				trchs = append(trchs, make(chan error, 1))
				go transceiver(c, []byte("TCP SERVER TEST"), trchs[i])
			}

			for _, ch := range trchs {
				for err := range ch {
					t.Errorf("#%d: %v", i, err)
				}
			}
			for _, ch := range tpchs {
				for err := range ch {
					t.Errorf("#%d: %v", i, err)
				}
			}
		})
	}
}

// TestUnixAndUnixpacketServer tests concurrent accept-read-write
// servers
func TestUnixAndUnixpacketServer(t *testing.T) {
	var unixAndUnixpacketServerTests = []struct {
		network, address string
	}{
		{"unix", testUnixAddr(t)},
		{"unix", "@nettest/go/unix"},

		{"unixpacket", testUnixAddr(t)},
		{"unixpacket", "@nettest/go/unixpacket"},
	}

	const N = 3

	for i, tt := range unixAndUnixpacketServerTests {
		if !testableListenArgs(tt.network, tt.address, "") {
			t.Logf("skipping %s test", tt.network+" "+tt.address)
			continue
		}

		ln, err := Listen(tt.network, tt.address)
		if err != nil {
			if perr := parseDialError(err); perr != nil {
				t.Error(perr)
			}
			t.Fatal(err)
		}

		var lss []*localServer
		var tpchs []chan error
		defer func() {
			for _, ls := range lss {
				ls.teardown()
			}
		}()
		for i := 0; i < N; i++ {
			ls := (&streamListener{Listener: ln}).newLocalServer()
			lss = append(lss, ls)
			tpchs = append(tpchs, make(chan error, 1))
		}
		for i := 0; i < N; i++ {
			ch := tpchs[i]
			handler := func(ls *localServer, ln Listener) { ls.transponder(ln, ch) }
			if err := lss[i].buildup(handler); err != nil {
				t.Fatal(err)
			}
		}

		var trchs []chan error
		for i := 0; i < N; i++ {
			d := Dialer{Timeout: someTimeout}
			c, err := d.Dial(lss[i].Listener.Addr().Network(), lss[i].Listener.Addr().String())
			if err != nil {
				if perr := parseDialError(err); perr != nil {
					t.Error(perr)
				}
				t.Fatal(err)
			}

			if addr := c.LocalAddr(); addr != nil {
				t.Logf("connected %s->%s", addr, lss[i].Listener.Addr())
			}

			defer c.Close()
			trchs = append(trchs, make(chan error, 1))
			go transceiver(c, []byte("UNIX AND UNIXPACKET SERVER TEST"), trchs[i])
		}

		for _, ch := range trchs {
			for err := range ch {
				t.Errorf("#%d: %v", i, err)
			}
		}
		for _, ch := range tpchs {
			for err := range ch {
				t.Errorf("#%d: %v", i, err)
			}
		}
	}
}

var udpServerTests = []struct {
	snet, saddr string // server endpoint
	tnet, taddr string // target endpoint for client
	dial        bool   // test with Dial
}{
	{snet: "udp", saddr: ":0", tnet: "udp", taddr: "127.0.0.1"},
	{snet: "udp", saddr: "0.0.0.0:0", tnet: "udp", taddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]:0", tnet: "udp", taddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::]:0", tnet: "udp", taddr: "::1"},

	{snet: "udp", saddr: ":0", tnet: "udp", taddr: "::1"},
	{snet: "udp", saddr: "0.0.0.0:0", tnet: "udp", taddr: "::1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]:0", tnet: "udp", taddr: "::1"},
	{snet: "udp", saddr: "[::]:0", tnet: "udp", taddr: "127.0.0.1"},

	{snet: "udp", saddr: ":0", tnet: "udp4", taddr: "127.0.0.1"},
	{snet: "udp", saddr: "0.0.0.0:0", tnet: "udp4", taddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]:0", tnet: "udp4", taddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::]:0", tnet: "udp6", taddr: "::1"},

	{snet: "udp", saddr: ":0", tnet: "udp6", taddr: "::1"},
	{snet: "udp", saddr: "0.0.0.0:0", tnet: "udp6", taddr: "::1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]:0", tnet: "udp6", taddr: "::1"},
	{snet: "udp", saddr: "[::]:0", tnet: "udp4", taddr: "127.0.0.1"},

	{snet: "udp", saddr: "127.0.0.1:0", tnet: "udp", taddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:127.0.0.1]:0", tnet: "udp", taddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::1]:0", tnet: "udp", taddr: "::1"},

	{snet: "udp4", saddr: ":0", tnet: "udp4", taddr: "127.0.0.1"},
	{snet: "udp4", saddr: "0.0.0.0:0", tnet: "udp4", taddr: "127.0.0.1"},
	{snet: "udp4", saddr: "[::ffff:0.0.0.0]:0", tnet: "udp4", taddr: "127.0.0.1"},

	{snet: "udp4", saddr: "127.0.0.1:0", tnet: "udp4", taddr: "127.0.0.1"},

	{snet: "udp6", saddr: ":0", tnet: "udp6", taddr: "::1"},
	{snet: "udp6", saddr: "[::]:0", tnet: "udp6", taddr: "::1"},

	{snet: "udp6", saddr: "[::1]:0", tnet: "udp6", taddr: "::1"},

	{snet: "udp", saddr: "127.0.0.1:0", tnet: "udp", taddr: "127.0.0.1", dial: true},

	{snet: "udp", saddr: "[::1]:0", tnet: "udp", taddr: "::1", dial: true},
}

func TestUDPServer(t *testing.T) {
	for i, tt := range udpServerTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			if !testableListenArgs(tt.snet, tt.saddr, tt.taddr) {
				t.Skipf("skipping %s %s<-%s test", tt.snet, tt.saddr, tt.taddr)
			}
			t.Logf("%s %s<-%s", tt.snet, tt.saddr, tt.taddr)

			c1, err := ListenPacket(tt.snet, tt.saddr)
			if err != nil {
				if perr := parseDialError(err); perr != nil {
					t.Error(perr)
				}
				t.Fatal(err)
			}

			ls := (&packetListener{PacketConn: c1}).newLocalServer()
			defer ls.teardown()
			tpch := make(chan error, 1)
			handler := func(ls *localPacketServer, c PacketConn) { packetTransponder(c, tpch) }
			if err := ls.buildup(handler); err != nil {
				t.Fatal(err)
			}

			trch := make(chan error, 1)
			_, port, err := SplitHostPort(ls.PacketConn.LocalAddr().String())
			if err != nil {
				t.Fatal(err)
			}
			if tt.dial {
				d := Dialer{Timeout: someTimeout}
				c2, err := d.Dial(tt.tnet, JoinHostPort(tt.taddr, port))
				if err != nil {
					if perr := parseDialError(err); perr != nil {
						t.Error(perr)
					}
					t.Fatal(err)
				}
				defer c2.Close()
				go transceiver(c2, []byte("UDP SERVER TEST"), trch)
			} else {
				c2, err := ListenPacket(tt.tnet, JoinHostPort(tt.taddr, "0"))
				if err != nil {
					if perr := parseDialError(err); perr != nil {
						t.Error(perr)
					}
					t.Fatal(err)
				}
				defer c2.Close()
				dst, err := ResolveUDPAddr(tt.tnet, JoinHostPort(tt.taddr, port))
				if err != nil {
					t.Fatal(err)
				}
				go packetTransceiver(c2, []byte("UDP SERVER TEST"), dst, trch)
			}

			for trch != nil || tpch != nil {
				select {
				case err, ok := <-trch:
					if !ok {
						trch = nil
					}
					if err != nil {
						t.Errorf("client: %v", err)
					}
				case err, ok := <-tpch:
					if !ok {
						tpch = nil
					}
					if err != nil {
						t.Errorf("server: %v", err)
					}
				}
			}
		})
	}
}

func TestUnixgramServer(t *testing.T) {
	var unixgramServerTests = []struct {
		saddr string // server endpoint
		caddr string // client endpoint
		dial  bool   // test with Dial
	}{
		{saddr: testUnixAddr(t), caddr: testUnixAddr(t)},
		{saddr: testUnixAddr(t), caddr: testUnixAddr(t), dial: true},

		{saddr: "@nettest/go/unixgram/server", caddr: "@nettest/go/unixgram/client"},
	}

	for i, tt := range unixgramServerTests {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			if !testableListenArgs("unixgram", tt.saddr, "") {
				t.Skipf("skipping unixgram %s<-%s test", tt.saddr, tt.caddr)
			}
			t.Logf("unixgram %s<-%s", tt.saddr, tt.caddr)

			c1, err := ListenPacket("unixgram", tt.saddr)
			if err != nil {
				if perr := parseDialError(err); perr != nil {
					t.Error(perr)
				}
				t.Fatal(err)
			}

			ls := (&packetListener{PacketConn: c1}).newLocalServer()
			defer ls.teardown()
			tpch := make(chan error, 1)
			handler := func(ls *localPacketServer, c PacketConn) { packetTransponder(c, tpch) }
			if err := ls.buildup(handler); err != nil {
				t.Fatal(err)
			}

			trch := make(chan error, 1)
			if tt.dial {
				d := Dialer{Timeout: someTimeout, LocalAddr: &UnixAddr{Net: "unixgram", Name: tt.caddr}}
				c2, err := d.Dial("unixgram", ls.PacketConn.LocalAddr().String())
				if err != nil {
					if perr := parseDialError(err); perr != nil {
						t.Error(perr)
					}
					t.Fatal(err)
				}
				defer os.Remove(c2.LocalAddr().String())
				defer c2.Close()
				go transceiver(c2, []byte(c2.LocalAddr().String()), trch)
			} else {
				c2, err := ListenPacket("unixgram", tt.caddr)
				if err != nil {
					if perr := parseDialError(err); perr != nil {
						t.Error(perr)
					}
					t.Fatal(err)
				}
				defer os.Remove(c2.LocalAddr().String())
				defer c2.Close()
				go packetTransceiver(c2, []byte("UNIXGRAM SERVER TEST"), ls.PacketConn.LocalAddr(), trch)
			}

			for trch != nil || tpch != nil {
				select {
				case err, ok := <-trch:
					if !ok {
						trch = nil
					}
					if err != nil {
						t.Errorf("client: %v", err)
					}
				case err, ok := <-tpch:
					if !ok {
						tpch = nil
					}
					if err != nil {
						t.Errorf("server: %v", err)
					}
				}
			}
		})
	}
}
