// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"errors"
	"fmt"
	"internal/testenv"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"
)

// testUnixAddr uses os.MkdirTemp to get a name that is unique.
func testUnixAddr(t testing.TB) string {
	// Pass an empty pattern to get a directory name that is as short as possible.
	// If we end up with a name longer than the sun_path field in the sockaddr_un
	// struct, we won't be able to make the syscall to open the socket.
	d, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(d); err != nil {
			t.Error(err)
		}
	})
	return filepath.Join(d, "sock")
}

func newLocalListener(t testing.TB, network string, lcOpt ...*ListenConfig) Listener {
	var lc *ListenConfig
	switch len(lcOpt) {
	case 0:
		lc = new(ListenConfig)
	case 1:
		lc = lcOpt[0]
	default:
		t.Helper()
		t.Fatal("too many ListenConfigs passed to newLocalListener: want 0 or 1")
	}

	listen := func(net, addr string) Listener {
		ln, err := lc.Listen(context.Background(), net, addr)
		if err != nil {
			t.Helper()
			t.Fatal(err)
		}
		return ln
	}

	switch network {
	case "tcp":
		if supportsIPv4() {
			return listen("tcp4", "127.0.0.1:0")
		}
		if supportsIPv6() {
			return listen("tcp6", "[::1]:0")
		}
	case "tcp4":
		if supportsIPv4() {
			return listen("tcp4", "127.0.0.1:0")
		}
	case "tcp6":
		if supportsIPv6() {
			return listen("tcp6", "[::1]:0")
		}
	case "unix", "unixpacket":
		return listen(network, testUnixAddr(t))
	}

	t.Helper()
	t.Fatalf("%s is not supported", network)
	return nil
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
	cl   []Conn    // accepted connection list
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
	defer ls.lnmu.Unlock()
	if ls.Listener != nil {
		network := ls.Listener.Addr().Network()
		address := ls.Listener.Addr().String()
		ls.Listener.Close()
		for _, c := range ls.cl {
			if err := c.Close(); err != nil {
				return err
			}
		}
		<-ls.done
		ls.Listener = nil
		switch network {
		case "unix", "unixpacket":
			os.Remove(address)
		}
	}
	return nil
}

func newLocalServer(t testing.TB, network string) *localServer {
	t.Helper()
	ln := newLocalListener(t, network)
	return &localServer{Listener: ln, done: make(chan bool)}
}

type streamListener struct {
	network, address string
	Listener
	done chan bool // signal that indicates server stopped
}

func (sl *streamListener) newLocalServer() *localServer {
	return &localServer{Listener: sl.Listener, done: make(chan bool)}
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

func (ls *localServer) transponder(ln Listener, ch chan<- error) {
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
	ls.cl = append(ls.cl, c)

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

func newLocalPacketListener(t testing.TB, network string, lcOpt ...*ListenConfig) PacketConn {
	var lc *ListenConfig
	switch len(lcOpt) {
	case 0:
		lc = new(ListenConfig)
	case 1:
		lc = lcOpt[0]
	default:
		t.Helper()
		t.Fatal("too many ListenConfigs passed to newLocalListener: want 0 or 1")
	}

	listenPacket := func(net, addr string) PacketConn {
		c, err := lc.ListenPacket(context.Background(), net, addr)
		if err != nil {
			t.Helper()
			t.Fatal(err)
		}
		return c
	}

	t.Helper()
	switch network {
	case "udp":
		if supportsIPv4() {
			return listenPacket("udp4", "127.0.0.1:0")
		}
		if supportsIPv6() {
			return listenPacket("udp6", "[::1]:0")
		}
	case "udp4":
		if supportsIPv4() {
			return listenPacket("udp4", "127.0.0.1:0")
		}
	case "udp6":
		if supportsIPv6() {
			return listenPacket("udp6", "[::1]:0")
		}
	case "unixgram":
		return listenPacket(network, testUnixAddr(t))
	}

	t.Fatalf("%s is not supported", network)
	return nil
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

func newLocalPacketServer(t testing.TB, network string) *localPacketServer {
	t.Helper()
	c := newLocalPacketListener(t, network)
	return &localPacketServer{PacketConn: c, done: make(chan bool)}
}

type packetListener struct {
	PacketConn
}

func (pl *packetListener) newLocalServer() *localPacketServer {
	return &localPacketServer{PacketConn: pl.PacketConn, done: make(chan bool)}
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

func spawnTestSocketPair(t testing.TB, net string) (client, server Conn) {
	t.Helper()

	ln := newLocalListener(t, net)
	defer ln.Close()
	var cerr, serr error
	acceptDone := make(chan struct{})
	go func() {
		server, serr = ln.Accept()
		acceptDone <- struct{}{}
	}()
	client, cerr = Dial(ln.Addr().Network(), ln.Addr().String())
	<-acceptDone
	if cerr != nil {
		if server != nil {
			server.Close()
		}
		t.Fatal(cerr)
	}
	if serr != nil {
		if client != nil {
			client.Close()
		}
		t.Fatal(serr)
	}
	return client, server
}

func startTestSocketPeer(t testing.TB, conn Conn, op string, chunkSize, totalSize int) (func(t testing.TB), error) {
	t.Helper()

	if runtime.GOOS == "windows" {
		// TODO(panjf2000): Windows has not yet implemented FileConn,
		//		remove this when it's implemented in https://go.dev/issues/9503.
		t.Fatalf("startTestSocketPeer is not supported on %s", runtime.GOOS)
	}

	f, err := conn.(interface{ File() (*os.File, error) }).File()
	if err != nil {
		return nil, err
	}

	cmd := testenv.Command(t, os.Args[0])
	cmd.Env = []string{
		"GO_NET_TEST_TRANSFER=1",
		"GO_NET_TEST_TRANSFER_OP=" + op,
		"GO_NET_TEST_TRANSFER_CHUNK_SIZE=" + strconv.Itoa(chunkSize),
		"GO_NET_TEST_TRANSFER_TOTAL_SIZE=" + strconv.Itoa(totalSize),
		"TMPDIR=" + os.Getenv("TMPDIR"),
	}
	cmd.ExtraFiles = append(cmd.ExtraFiles, f)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	cmdCh := make(chan error, 1)
	go func() {
		err := cmd.Wait()
		conn.Close()
		f.Close()
		cmdCh <- err
	}()

	return func(tb testing.TB) {
		err := <-cmdCh
		if err != nil {
			tb.Errorf("process exited with error: %v", err)
		}
	}, nil
}

func init() {
	if os.Getenv("GO_NET_TEST_TRANSFER") == "" {
		return
	}
	defer os.Exit(0)

	f := os.NewFile(uintptr(3), "splice-test-conn")
	defer f.Close()

	conn, err := FileConn(f)
	if err != nil {
		log.Fatal(err)
	}

	var chunkSize int
	if chunkSize, err = strconv.Atoi(os.Getenv("GO_NET_TEST_TRANSFER_CHUNK_SIZE")); err != nil {
		log.Fatal(err)
	}
	buf := make([]byte, chunkSize)

	var totalSize int
	if totalSize, err = strconv.Atoi(os.Getenv("GO_NET_TEST_TRANSFER_TOTAL_SIZE")); err != nil {
		log.Fatal(err)
	}

	var fn func([]byte) (int, error)
	switch op := os.Getenv("GO_NET_TEST_TRANSFER_OP"); op {
	case "r":
		fn = conn.Read
	case "w":
		defer conn.Close()

		fn = conn.Write
	default:
		log.Fatalf("unknown op %q", op)
	}

	var n int
	for count := 0; count < totalSize; count += n {
		if count+chunkSize > totalSize {
			buf = buf[:totalSize-count]
		}

		var err error
		if n, err = fn(buf); err != nil {
			return
		}
	}
}
