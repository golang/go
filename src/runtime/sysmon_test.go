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
	ffn("dialed %d times within %v", srv.bucketTotal, srv.end.Sub(srv.start))
	ffn("timeBucket\tcount\tpercent")
	for bucket, cnt := range srv.bucket {
		percent := float64(cnt) / float64(srv.bucketTotal) * 100.0
		if bucket == len(srv.bucket)-1 {
			ffn("[%2d, ~)ms\t%d\t%.2f%%", bucket, cnt, percent)
		} else {
			ffn("[%2d,%2d)ms\t%d\t%.2f%%", bucket, bucket+1, cnt, percent)
		}
	}
}

func TestSysmonReadyNetpollWaitersASAP(t *testing.T) {
	if runtime.GOOS == "netbsd" && runtime.NeedSysmonWorkaround {
		t.Skip("netbsd 9.2 earlier")
	}
	if runtime.GOARCH == "wasm" {
		t.Skip("no sysmon on wasm yet")
	}
	if runtime.GOOS == "openbsd" {
		// usleep(20us) actually slept 20ms. see issue #17712.
		t.Skip("sysmon may oversleep on openbsd.")
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

	// expect more than 80% dialings accomplished within 2ms in general.
	// but android emulator may be slow, so more patience needed.
	bucket, percent := 2, 80.0
	if runtime.GOOS == "android" {
		bucket = 9
	}
	if !srv.expect(bucket, percent) {
		t.Fail()
	}
	srv.printf(t.Logf)
}
