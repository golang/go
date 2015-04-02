// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"os"
	"sync"
)

func newLocalListener(network string) (Listener, error) {
	switch network {
	case "tcp", "tcp4", "tcp6":
		if supportsIPv4 {
			return Listen("tcp4", "127.0.0.1:0")
		}
		if supportsIPv6 {
			return Listen("tcp6", "[::1]:0")
		}
	case "unix", "unixpacket":
		return Listen(network, testUnixAddr())
	}
	return nil, fmt.Errorf("%s is not supported", network)
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

func (dss *dualStackServer) putConn(c Conn) error {
	dss.cmu.Lock()
	dss.cs = append(dss.cs, c)
	dss.cmu.Unlock()
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

func newDualStackServer(lns []streamListener) (*dualStackServer, error) {
	dss := &dualStackServer{lns: lns, port: "0"}
	for i := range dss.lns {
		ln, err := Listen(dss.lns[i].network, JoinHostPort(dss.lns[i].address, dss.port))
		if err != nil {
			for _, ln := range dss.lns {
				ln.Listener.Close()
			}
			return nil, err
		}
		dss.lns[i].Listener = ln
		dss.lns[i].done = make(chan bool)
		if dss.port == "0" {
			if _, dss.port, err = SplitHostPort(ln.Addr().String()); err != nil {
				for _, ln := range dss.lns {
					ln.Listener.Close()
				}
				return nil, err
			}
		}
	}
	return dss, nil
}
