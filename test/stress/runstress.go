// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The runstress tool stresses the runtime.
//
// It runs forever and should never fail. It tries to stress the garbage collector,
// maps, channels, the network, and everything else provided by the runtime.
package main

import (
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strconv"
	"time"
)

var (
	v         = flag.Bool("v", false, "verbose")
	doMaps    = flag.Bool("maps", true, "stress maps")
	doExec    = flag.Bool("exec", true, "stress exec")
	doChan    = flag.Bool("chan", true, "stress channels")
	doNet     = flag.Bool("net", true, "stress networking")
	doParseGo = flag.Bool("parsego", true, "stress parsing Go (generates garbage)")
)

func Println(a ...interface{}) {
	if *v {
		log.Println(a...)
	}
}

func dialStress(a net.Addr) {
	for {
		d := net.Dialer{Timeout: time.Duration(rand.Intn(1e9))}
		c, err := d.Dial("tcp", a.String())
		if err == nil {
			Println("did dial")
			go func() {
				time.Sleep(time.Duration(rand.Intn(500)) * time.Millisecond)
				c.Close()
				Println("closed dial")
			}()
		}
		// Don't run out of ephermeral ports too quickly:
		time.Sleep(250 * time.Millisecond)
	}
}

func stressNet() {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		size, _ := strconv.Atoi(r.FormValue("size"))
		w.Write(make([]byte, size))
	}))
	go dialStress(ts.Listener.Addr())
	for {
		size := rand.Intn(128 << 10)
		res, err := http.Get(fmt.Sprintf("%s/?size=%d", ts.URL, size))
		if err != nil {
			log.Fatalf("stressNet: http Get error: %v", err)
		}
		if res.StatusCode != 200 {
			log.Fatalf("stressNet: Status code = %d", res.StatusCode)
		}
		n, err := io.Copy(ioutil.Discard, res.Body)
		if err != nil {
			log.Fatalf("stressNet: io.Copy: %v", err)
		}
		if n != int64(size) {
			log.Fatalf("stressNet: copied = %d; want %d", n, size)
		}
		res.Body.Close()
		Println("did http", size)
	}
}

func doAnExec() {
	exit := rand.Intn(2)
	wantOutput := fmt.Sprintf("output-%d", rand.Intn(1e9))
	cmd := exec.Command("/bin/sh", "-c", fmt.Sprintf("echo %s; exit %d", wantOutput, exit))
	out, err := cmd.CombinedOutput()
	if exit == 1 {
		if err == nil {
			log.Fatal("stressExec: unexpected exec success")
		}
		return
	}
	if err != nil {
		log.Fatalf("stressExec: exec failure: %v: %s", err, out)
	}
	wantOutput += "\n"
	if string(out) != wantOutput {
		log.Fatalf("stressExec: exec output = %q; want %q", out, wantOutput)
	}
	Println("did exec")
}

func stressExec() {
	gate := make(chan bool, 10) // max execs at once
	for {
		gate <- true
		go func() {
			doAnExec()
			<-gate
		}()
	}
}

func ringf(in <-chan int, out chan<- int, donec chan<- bool) {
	for {
		n := <-in
		if n == 0 {
			donec <- true
			return
		}
		out <- n - 1
	}
}

func threadRing(bufsize int) {
	const N = 100
	donec := make(chan bool)
	one := make(chan int, bufsize) // will be input to thread 1
	var in, out chan int = nil, one
	for i := 1; i <= N-1; i++ {
		in, out = out, make(chan int, bufsize)
		go ringf(in, out, donec)
	}
	go ringf(out, one, donec)
	one <- N
	<-donec
	Println("did threadring of", bufsize)
}

func stressChannels() {
	for {
		threadRing(0)
		threadRing(1)
	}
}

func main() {
	flag.Parse()
	for want, f := range map[*bool]func(){
		doMaps:    stressMaps,
		doNet:     stressNet,
		doExec:    stressExec,
		doChan:    stressChannels,
		doParseGo: stressParseGo,
	} {
		if *want {
			go f()
		}
	}
	select {}
}
