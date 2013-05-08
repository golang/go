// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows plan9

package net

import (
	"time"
)

var testingIssue5349 bool // used during tests

// resolveAndDialChannel is the simple pure-Go implementation of
// resolveAndDial, still used on operating systems where the deadline
// hasn't been pushed down into the pollserver. (Plan 9 and some old
// versions of Windows)
func resolveAndDialChannel(net, addr string, localAddr Addr, deadline time.Time) (Conn, error) {
	var timeout time.Duration
	if !deadline.IsZero() {
		timeout = deadline.Sub(time.Now())
	}
	if timeout <= 0 {
		ra, err := resolveAddr("dial", net, addr, noDeadline)
		if err != nil {
			return nil, err
		}
		return dial(net, addr, localAddr, ra, noDeadline)
	}
	t := time.NewTimer(timeout)
	defer t.Stop()
	type pair struct {
		Conn
		error
	}
	ch := make(chan pair, 1)
	resolvedAddr := make(chan Addr, 1)
	go func() {
		if testingIssue5349 {
			time.Sleep(time.Millisecond)
		}
		ra, err := resolveAddr("dial", net, addr, noDeadline)
		if err != nil {
			ch <- pair{nil, err}
			return
		}
		resolvedAddr <- ra // in case we need it for OpError
		c, err := dial(net, addr, localAddr, ra, noDeadline)
		ch <- pair{c, err}
	}()
	select {
	case <-t.C:
		// Try to use the real Addr in our OpError, if we resolved it
		// before the timeout. Otherwise we just use stringAddr.
		var ra Addr
		select {
		case a := <-resolvedAddr:
			ra = a
		default:
			ra = &stringAddr{net, addr}
		}
		err := &OpError{
			Op:   "dial",
			Net:  net,
			Addr: ra,
			Err:  &timeoutError{},
		}
		return nil, err
	case p := <-ch:
		return p.Conn, p.error
	}
}
