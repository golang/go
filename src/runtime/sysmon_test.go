// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test netpoll waiters G's will be unparked as soon as IO readied
// even though M's are busy with G's in local runq.

package runtime_test

import (
	"net"
	"runtime"
	"testing"
	"time"
)

type busysrv struct {
	l           net.Listener
	bucket      []int
	bucketTotal int
	exit        chan struct{}
	start, end  time.Time
}

func (srv *busysrv) stop() {
	close(srv.exit)
}

func (srv *busysrv) startListening() {
	l, _ := net.Listen("tcp4", "localhost:0")
	bucket := make([]int, 12)
	exit := make(chan struct{})
	srv.l = l
	srv.bucket = bucket
	srv.exit = exit
	go func() {
		for {
			select {
			case _, ok := <-exit:
				if !ok {
					l.Close()
					return
				}
			default:
			}

			if con, _ := l.Accept(); con != nil {
				con.Close()
			}
		}
	}()
}

func (srv *busysrv) startDialing() {
	srv.start = time.Now()
	defer func() {
		srv.end = time.Now()
	}()
	network, addr := srv.l.Addr().Network(), srv.l.Addr().String()
	for {
		select {
		case _, ok := <-srv.exit:
			if !ok {
				return
			}
		default:
		}

		start := time.Now()
		con, _ := net.Dial(network, addr)
		ms := int(time.Since(start) / 1000000)
		if ms >= len(srv.bucket) {
			ms = len(srv.bucket) - 1
		}
		srv.bucket[ms]++
		srv.bucketTotal++
		if con != nil {
			con.Close()
		}
	}
}

func (srv *busysrv) busy() {
	for {
		select {
		case _, ok := <-srv.exit:
			if !ok {
				return
			}
		default:
		}
		runtime.Goyield() // simulate many runnable G's in local runq.
	}
}

func (srv *busysrv) expect(bucket int, percent float64) bool {
	count := 0
	for i := 0; i < bucket && i < len(srv.bucket); i++ {
		count += srv.bucket[i]
	}
	return float64(count)/float64(srv.bucketTotal)*100.0 > percent
}

func (srv *busysrv) printf(ffn func(format string, args ...interface{})) {
	ffn("dialed %d times within %v\n", srv.bucketTotal, srv.end.Sub(srv.start))
	ffn("timeBucket\tcount\tpercent\n")
	for ms, cnt := range srv.bucket {
		ffn("[%2d,%2d)ms\t%d\t%.2f%%\n", ms, ms+1, cnt, float64(cnt)/float64(srv.bucketTotal)*100.0)
	}
}

func TestSysmonReadyNetpollWaitersASAP(t *testing.T) {
	if runtime.GOOS == "netbsd" && runtime.NeedSysmonWorkaround {
		t.Skip("netbsd 9.2 earlier")
	}
	if runtime.GOARCH == "wasm" {
		t.Skip("no sysmon on wasm yet")
	}

	// sysmon may starve if host load is too high.
	np := runtime.GOMAXPROCS(0)
	if np >= runtime.NumCPU() {
		t.Skip("host load may be too high to run sysmon.")
	}

	srv := &busysrv{}
	srv.startListening()
	for i := 0; i < np*5; i++ {
		go srv.busy()
	}
	go srv.startDialing()

	time.Sleep(time.Second)
	srv.stop()
	time.Sleep(time.Millisecond * 100)

	// expect more than 80% dialings accomplished within 2ms.
	if !srv.expect(2, 80.0) {
		t.Fail()
	}
	srv.printf(t.Logf)
}
