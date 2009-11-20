// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a simple demo of Go running under Native Client.
// It is a tetris clone built on top of the exp/nacl/av and exp/draw
// packages.
package main

import (
	"exp/nacl/av";
	"exp/nacl/srpc";
	"log";
	"runtime";
	"os";
)

var sndc chan []uint16

func main() {
	// Native Client requires that some calls are issued
	// consistently by the same OS thread.
	runtime.LockOSThread();

	if srpc.Enabled() {
		go srpc.ServeRuntime()
	}

	args := os.Args;
	p := pieces4;
	if len(args) > 1 && args[1] == "-5" {
		p = pieces5
	}
	dx, dy := 500, 500;
	w, err := av.Init(av.SubsystemVideo|av.SubsystemAudio, dx, dy);
	if err != nil {
		log.Exit(err)
	}

	sndc = make(chan []uint16, 10);
	go audioServer();
	Play(p, w);
}

func audioServer() {
	// Native Client requires that all audio calls
	// original from a single OS thread.
	runtime.LockOSThread();

	n, err := av.AudioStream(nil);
	if err != nil {
		log.Exit(err)
	}
	for {
		b := <-sndc;
		for len(b)*2 >= n {
			var a []uint16;
			a, b = b[0:n/2], b[n/2:];
			n, err = av.AudioStream(a);
			if err != nil {
				log.Exit(err)
			}
			println(n, len(b)*2);
		}
		a := make([]uint16, n/2);
		for i := range b {
			a[i] = b[i]
		}
		n, err = av.AudioStream(a);
	}
}

func PlaySound(b []uint16)	{ sndc <- b }

var whoosh = []uint16{
// Insert your favorite sound samples here.
}
