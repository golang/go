// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows plan9

package net

import "time"

// dialChannel is the simple pure-Go implementation of dial, still
// used on operating systems where the deadline hasn't been pushed
// down into the pollserver. (Plan 9 and some old versions of Windows)
func dialChannel(net string, ra Addr, dialer func(time.Time) (Conn, error), deadline time.Time) (Conn, error) {
	if deadline.IsZero() {
		return dialer(noDeadline)
	}
	timeout := deadline.Sub(time.Now())
	if timeout <= 0 {
		return nil, &OpError{Op: "dial", Net: net, Source: nil, Addr: ra, Err: errTimeout}
	}
	t := time.NewTimer(timeout)
	defer t.Stop()
	type racer struct {
		Conn
		error
	}
	ch := make(chan racer, 1)
	go func() {
		testHookDialChannel()
		c, err := dialer(noDeadline)
		ch <- racer{c, err}
	}()
	select {
	case <-t.C:
		return nil, &OpError{Op: "dial", Net: net, Source: nil, Addr: ra, Err: errTimeout}
	case racer := <-ch:
		return racer.Conn, racer.error
	}
}
