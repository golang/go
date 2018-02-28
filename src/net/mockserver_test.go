// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"testing"
	"time"
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

func newLocalListener(network string) (Listener, error) {
	switch network {
	case "tcp":
		if supportsIPv4() {
			if ln, err := Listen("tcp4", "127.0.0.1:0"); err == nil {
				return ln, nil
			}
		}
		if supportsIPv6() {
			return Listen("tcp6", "[::1]:0")
		}
	case "tcp4":
		if supportsIPv4() {
			return Listen("tcp4", "127.0.0.1:0")
		}
	case "tcp6":
		if supportsIPv6() {
			return Listen("tcp6", "[::1]:0")
		}
	case "unix", "unixpacket":
		return Listen(network, testUnixAddr())
	}
	return nil, fmt.Errorf("%s is not supported", network)
}

func newDualStackListener() (lns []*TCPListener, err error) {
	var args = []struct {
		network string
		TCPAddr
	}{
		{"tcp4", TCPAddr{IP: IPv4(127, 0, 0, 1)}},
		{"tcp6", TCPAddr{IP: IPv6loopback}},
	}
	for i := 0; i < 64; i++ {
		var port int
		var lns []*TCPListener
		for _, arg := range args {
			arg.TCPAddr.Port = port
			ln, err := ListenTCP(arg.network, &arg.TCPAddr)
			if err != nil {
				continue
			}
			port = ln.Addr().(*TCPAddr).Port
			lns = append(lns, ln)
		}
		if len(lns) != len(args) {
			for _, ln := range lns {
				ln.Close()
			}
			continue
		}
		return lns, nil
	}
	return nil, errors.New("no dualstack port available")
}

type localServer struct {
	lnmu sync.RWMutex
	Listener
	done chan bool // signal that indicates server stopped
}

func (ls *localServer) buildup(handler func(*localServer, Listener)) error {
	go func() {
		handler(ls, ls.Listener)
		close(ls.done)
	}()
	return nil
}

func (ls *localServer) teardown() error {
	ls.lnmu.Lock()
	if ls.Listener != nil {
		network := ls.Listener.Addr().Network()
		address := ls.Listener.Addr().String()
		ls.Listener.Close()
		<-ls.done
		ls.Listener = nil
		switch network {
		case "unix", "unixpacket":
			os.Remove(address)
		}
	}
	ls.lnmu.Unlock()
	return nil
}

func newLocalServer(network string) (*localServer, error) {
	ln, err := newLocalListener(network)
	if err != nil {
		return nil, err
	}
	return &localServer{Listener: ln, done: make(chan bool)}, nil
}

type streamListener struct {
	network, address string
	Listener
	done chan bool // signal that indicates server stopped
}

func (sl *streamListener) newLocalServer() (*localServer, error) {
	return &localServer{Listener: sl.Listener, done: make(chan bool)}, nil
}

type dualStackServer struct {
	lnmu sync.RWMutex
	lns  []streamListener
	port string

	cmu sync.RWMutex
	cs  []Conn // established connections at the passive open side
}

func (dss *dualStackServer) buildup(handler func(*dualStackServer, Listener)) error {
	for i := range dss.lns {
		go func(i int) {
			handler(dss, dss.lns[i].Listener)
			close(dss.lns[i].done)
		}(i)
	}
	return nil
}

func (dss *dualStackServer) teardownNetwork(network string) error {
	dss.lnmu.Lock()
	for i := range dss.lns {
		if network == dss.lns[i].network && dss.lns[i].Listener != nil {
			dss.lns[i].Listener.Close()
			<-dss.lns[i].done
			dss.lns[i].Listener = nil
		}
	}
	dss.lnmu.Unlock()
	return nil
}

func (dss *dualStackServer) teardown() error {
	dss.lnmu.Lock()
	for i := range dss.lns {
		if dss.lns[i].Listener != nil {
			dss.lns[i].Listener.Close()
			<-dss.lns[i].done
		}
	}
	dss.lns = dss.lns[:0]
	dss.lnmu.Unlock()
	dss.cmu.Lock()
	for _, c := range dss.cs {
		c.Close()
	}
	dss.cs = dss.cs[:0]
	dss.cmu.Unlock()
	return nil
}

func newDualStackServer() (*dualStackServer, error) {
	lns, err := newDualStackListener()
	if err != nil {
		return nil, err
	}
	_, port, err := SplitHostPort(lns[0].Addr().String())
	if err != nil {
		lns[0].Close()
		lns[1].Close()
		return nil, err
	}
	return &dualStackServer{
		lns: []streamListener{
			{network: "tcp4", address: lns[0].Addr().String(), Listener: lns[0], done: make(chan bool)},
			{network: "tcp6", address: lns[1].Addr().String(), Listener: lns[1], done: make(chan bool)},
		},
		port: port,
	}, nil
}

func transponder(ln Listener, ch chan<- error) {
	defer close(ch)

	switch ln := ln.(type) {
	case *TCPListener:
		ln.SetDeadline(time.Now().Add(someTimeout))
	case *UnixListener:
		ln.SetDeadline(time.Now().Add(someTimeout))
	}
	c, err := ln.Accept()
	if err != nil {
		if perr := parseAcceptError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
	defer c.Close()

	network := ln.Addr().Network()
	if c.LocalAddr().Network() != network || c.RemoteAddr().Network() != network {
		ch <- fmt.Errorf("got %v->%v; expected %v->%v", c.LocalAddr().Network(), c.RemoteAddr().Network(), network, network)
		return
	}
	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))

	b := make([]byte, 256)
	n, err := c.Read(b)
	if err != nil {
		if perr := parseReadError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
	if _, err := c.Write(b[:n]); err != nil {
		if perr := parseWriteError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
}

func transceiver(c Conn, wb []byte, ch chan<- error) {
	defer close(ch)

	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))

	n, err := c.Write(wb)
	if err != nil {
		if perr := parseWriteError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
	if n != len(wb) {
		ch <- fmt.Errorf("wrote %d; want %d", n, len(wb))
	}
	rb := make([]byte, len(wb))
	n, err = c.Read(rb)
	if err != nil {
		if perr := parseReadError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
	if n != len(wb) {
		ch <- fmt.Errorf("read %d; want %d", n, len(wb))
	}
}

func timeoutReceiver(c Conn, d, min, max time.Duration, ch chan<- error) {
	var err error
	defer func() { ch <- err }()

	t0 := time.Now()
	if err = c.SetReadDeadline(time.Now().Add(d)); err != nil {
		return
	}
	b := make([]byte, 256)
	var n int
	n, err = c.Read(b)
	t1 := time.Now()
	if n != 0 || err == nil || !err.(Error).Timeout() {
		err = fmt.Errorf("Read did not return (0, timeout): (%d, %v)", n, err)
		return
	}
	if dt := t1.Sub(t0); min > dt || dt > max && !testing.Short() {
		err = fmt.Errorf("Read took %s; expected %s", dt, d)
		return
	}
}

func timeoutTransmitter(c Conn, d, min, max time.Duration, ch chan<- error) {
	var err error
	defer func() { ch <- err }()

	t0 := time.Now()
	if err = c.SetWriteDeadline(time.Now().Add(d)); err != nil {
		return
	}
	var n int
	for {
		n, err = c.Write([]byte("TIMEOUT TRANSMITTER"))
		if err != nil {
			break
		}
	}
	t1 := time.Now()
	if err == nil || !err.(Error).Timeout() {
		err = fmt.Errorf("Write did not return (any, timeout): (%d, %v)", n, err)
		return
	}
	if dt := t1.Sub(t0); min > dt || dt > max && !testing.Short() {
		err = fmt.Errorf("Write took %s; expected %s", dt, d)
		return
	}
}

func newLocalPacketListener(network string) (PacketConn, error) {
	switch network {
	case "udp":
		if supportsIPv4() {
			return ListenPacket("udp4", "127.0.0.1:0")
		}
		if supportsIPv6() {
			return ListenPacket("udp6", "[::1]:0")
		}
	case "udp4":
		if supportsIPv4() {
			return ListenPacket("udp4", "127.0.0.1:0")
		}
	case "udp6":
		if supportsIPv6() {
			return ListenPacket("udp6", "[::1]:0")
		}
	case "unixgram":
		return ListenPacket(network, testUnixAddr())
	}
	return nil, fmt.Errorf("%s is not supported", network)
}

func newDualStackPacketListener() (cs []*UDPConn, err error) {
	var args = []struct {
		network string
		UDPAddr
	}{
		{"udp4", UDPAddr{IP: IPv4(127, 0, 0, 1)}},
		{"udp6", UDPAddr{IP: IPv6loopback}},
	}
	for i := 0; i < 64; i++ {
		var port int
		var cs []*UDPConn
		for _, arg := range args {
			arg.UDPAddr.Port = port
			c, err := ListenUDP(arg.network, &arg.UDPAddr)
			if err != nil {
				continue
			}
			port = c.LocalAddr().(*UDPAddr).Port
			cs = append(cs, c)
		}
		if len(cs) != len(args) {
			for _, c := range cs {
				c.Close()
			}
			continue
		}
		return cs, nil
	}
	return nil, errors.New("no dualstack port available")
}

type localPacketServer struct {
	pcmu sync.RWMutex
	PacketConn
	done chan bool // signal that indicates server stopped
}

func (ls *localPacketServer) buildup(handler func(*localPacketServer, PacketConn)) error {
	go func() {
		handler(ls, ls.PacketConn)
		close(ls.done)
	}()
	return nil
}

func (ls *localPacketServer) teardown() error {
	ls.pcmu.Lock()
	if ls.PacketConn != nil {
		network := ls.PacketConn.LocalAddr().Network()
		address := ls.PacketConn.LocalAddr().String()
		ls.PacketConn.Close()
		<-ls.done
		ls.PacketConn = nil
		switch network {
		case "unixgram":
			os.Remove(address)
		}
	}
	ls.pcmu.Unlock()
	return nil
}

func newLocalPacketServer(network string) (*localPacketServer, error) {
	c, err := newLocalPacketListener(network)
	if err != nil {
		return nil, err
	}
	return &localPacketServer{PacketConn: c, done: make(chan bool)}, nil
}

type packetListener struct {
	PacketConn
}

func (pl *packetListener) newLocalServer() (*localPacketServer, error) {
	return &localPacketServer{PacketConn: pl.PacketConn, done: make(chan bool)}, nil
}

func packetTransponder(c PacketConn, ch chan<- error) {
	defer close(ch)

	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))

	b := make([]byte, 256)
	n, peer, err := c.ReadFrom(b)
	if err != nil {
		if perr := parseReadError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
	if peer == nil { // for connected-mode sockets
		switch c.LocalAddr().Network() {
		case "udp":
			peer, err = ResolveUDPAddr("udp", string(b[:n]))
		case "unixgram":
			peer, err = ResolveUnixAddr("unixgram", string(b[:n]))
		}
		if err != nil {
			ch <- err
			return
		}
	}
	if _, err := c.WriteTo(b[:n], peer); err != nil {
		if perr := parseWriteError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
}

func packetTransceiver(c PacketConn, wb []byte, dst Addr, ch chan<- error) {
	defer close(ch)

	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))

	n, err := c.WriteTo(wb, dst)
	if err != nil {
		if perr := parseWriteError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
	if n != len(wb) {
		ch <- fmt.Errorf("wrote %d; want %d", n, len(wb))
	}
	rb := make([]byte, len(wb))
	n, _, err = c.ReadFrom(rb)
	if err != nil {
		if perr := parseReadError(err); perr != nil {
			ch <- perr
		}
		ch <- err
		return
	}
	if n != len(wb) {
		ch <- fmt.Errorf("read %d; want %d", n, len(wb))
	}
}

func timeoutPacketReceiver(c PacketConn, d, min, max time.Duration, ch chan<- error) {
	var err error
	defer func() { ch <- err }()

	t0 := time.Now()
	if err = c.SetReadDeadline(time.Now().Add(d)); err != nil {
		return
	}
	b := make([]byte, 256)
	var n int
	n, _, err = c.ReadFrom(b)
	t1 := time.Now()
	if n != 0 || err == nil || !err.(Error).Timeout() {
		err = fmt.Errorf("ReadFrom did not return (0, timeout): (%d, %v)", n, err)
		return
	}
	if dt := t1.Sub(t0); min > dt || dt > max && !testing.Short() {
		err = fmt.Errorf("ReadFrom took %s; expected %s", dt, d)
		return
	}
}
