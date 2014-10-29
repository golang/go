// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"fmt"
	"testing"
	"time"
)

var testDNSFlood = flag.Bool("dnsflood", false, "whether to test dns query flooding")

func TestDNSThreadLimit(t *testing.T) {
	if !*testDNSFlood {
		t.Skip("test disabled; use -dnsflood to enable")
	}

	const N = 10000
	c := make(chan int, N)
	for i := 0; i < N; i++ {
		go func(i int) {
			LookupIP(fmt.Sprintf("%d.net-test.golang.org", i))
			c <- 1
		}(i)
	}
	// Don't bother waiting for the stragglers; stop at 0.9 N.
	for i := 0; i < N*9/10; i++ {
		if i%100 == 0 {
			//println("TestDNSThreadLimit:", i)
		}
		<-c
	}

	// If we're still here, it worked.
}

func TestLookupIPDeadline(t *testing.T) {
	if !*testDNSFlood {
		t.Skip("test disabled; use -dnsflood to enable")
	}

	const N = 5000
	const timeout = 3 * time.Second
	c := make(chan error, 2*N)
	for i := 0; i < N; i++ {
		name := fmt.Sprintf("%d.net-test.golang.org", i)
		go func() {
			_, err := lookupIPDeadline(name, time.Now().Add(timeout/2))
			c <- err
		}()
		go func() {
			_, err := lookupIPDeadline(name, time.Now().Add(timeout))
			c <- err
		}()
	}
	qstats := struct {
		succeeded, failed         int
		timeout, temporary, other int
		unknown                   int
	}{}
	deadline := time.After(timeout + time.Second)
	for i := 0; i < 2*N; i++ {
		select {
		case <-deadline:
			t.Fatal("deadline exceeded")
		case err := <-c:
			switch err := err.(type) {
			case nil:
				qstats.succeeded++
			case Error:
				qstats.failed++
				if err.Timeout() {
					qstats.timeout++
				}
				if err.Temporary() {
					qstats.temporary++
				}
				if !err.Timeout() && !err.Temporary() {
					qstats.other++
				}
			default:
				qstats.failed++
				qstats.unknown++
			}
		}
	}

	// A high volume of DNS queries for sub-domain of golang.org
	// would be coordinated by authoritative or recursive server,
	// or stub resolver which implements query-response rate
	// limitation, so we can expect some query successes and more
	// failures including timeout, temporary and other here.
	// As a rule, unknown must not be shown but it might possibly
	// happen due to issue 4856 for now.
	t.Logf("%v succeeded, %v failed (%v timeout, %v temporary, %v other, %v unknown)", qstats.succeeded, qstats.failed, qstats.timeout, qstats.temporary, qstats.other, qstats.unknown)
}
