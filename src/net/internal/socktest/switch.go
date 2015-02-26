// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package socktest provides utilities for socket testing.
package socktest

import "sync"

func switchInit(sw *Switch) {
	sw.fltab = make(map[FilterType]Filter)
	sw.sotab = make(Sockets)
	sw.stats = make(stats)
}

// A Switch represents a callpath point switch for socket system
// calls.
type Switch struct {
	once sync.Once

	fmu   sync.RWMutex
	fltab map[FilterType]Filter

	smu   sync.RWMutex
	sotab Sockets
	stats stats
}

// Stats returns a list of per-cookie socket statistics.
func (sw *Switch) Stats() []Stat {
	var st []Stat
	sw.smu.RLock()
	for _, s := range sw.stats {
		ns := *s
		st = append(st, ns)
	}
	sw.smu.RUnlock()
	return st
}

// Sockets returns mappings of socket descriptor to socket status.
func (sw *Switch) Sockets() Sockets {
	sw.smu.RLock()
	tab := make(Sockets, len(sw.sotab))
	for i, s := range sw.sotab {
		tab[i] = s
	}
	sw.smu.RUnlock()
	return tab
}

// A Cookie represents a 3-tuple of a socket; address family, socket
// type and protocol number.
type Cookie uint64

// Family returns an address family.
func (c Cookie) Family() int { return int(c >> 48) }

// Type returns a socket type.
func (c Cookie) Type() int { return int(c << 16 >> 32) }

// Protocol returns a protocol number.
func (c Cookie) Protocol() int { return int(c & 0xff) }

func cookie(family, sotype, proto int) Cookie {
	return Cookie(family)<<48 | Cookie(sotype)&0xffffffff<<16 | Cookie(proto)&0xff
}

// A Status represents the status of a socket.
type Status struct {
	Cookie    Cookie
	Err       error // error status of socket system call
	SocketErr int   // error status of socket by SO_ERROR
}

// A Stat represents a per-cookie socket statistics.
type Stat struct {
	Family   int // address family
	Type     int // socket type
	Protocol int // protocol number

	Opened    uint64 // number of sockets opened
	Accepted  uint64 // number of sockets accepted
	Connected uint64 // number of sockets connected
	Closed    uint64 // number of sockets closed
}

type stats map[Cookie]*Stat

func (st stats) getLocked(c Cookie) *Stat {
	s, ok := st[c]
	if !ok {
		s = &Stat{Family: c.Family(), Type: c.Type(), Protocol: c.Protocol()}
		st[c] = s
	}
	return s
}

// A FilterType represents a filter type.
type FilterType int

const (
	FilterSocket        FilterType = iota // for Socket
	FilterAccept                          // for Accept or Accept4
	FilterConnect                         // for Connect or ConnectEx
	FilterGetsockoptInt                   // for GetsockoptInt
	FilterClose                           // for Close or Closesocket
)

// A Filter represents a socket system call filter.
//
// It will only be executed before a system call for a socket that has
// an entry in internal table.
// If the filter returns a non-nil error, the execution of system call
// will be canceled and the system call function returns the non-nil
// error.
// It can return a non-nil AfterFilter for filtering after the
// execution of the system call.
type Filter func(*Status) (AfterFilter, error)

func (f Filter) apply(st *Status) (AfterFilter, error) {
	if f == nil {
		return nil, nil
	}
	return f(st)
}

// An AfterFilter represents a socket system call filter after an
// execution of a system call.
//
// It will only be executed after a system call for a socket that has
// an entry in internal table.
// If the filter returns a non-nil error, the system call function
// returns the non-nil error.
type AfterFilter func(*Status) error

func (f AfterFilter) apply(st *Status) error {
	if f == nil {
		return nil
	}
	return f(st)
}

// Set deploys the socket system call filter f for the filter type t.
func (sw *Switch) Set(t FilterType, f Filter) {
	sw.once.Do(func() { switchInit(sw) })
	sw.fmu.Lock()
	sw.fltab[t] = f
	sw.fmu.Unlock()
}
