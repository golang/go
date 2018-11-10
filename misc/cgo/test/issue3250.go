// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package cgotest

/*
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

static void *thread(void *p) {
	const int M = 100;
	int i;
	(void)p;
	for (i = 0; i < M; i++) {
		pthread_kill(pthread_self(), SIGCHLD);
		usleep(rand() % 20 + 5);
	}
	return NULL;
}
void testSendSIG() {
	const int N = 20;
	int i;
	pthread_t tid[N];
	for (i = 0; i < N; i++) {
		usleep(rand() % 200 + 100);
		pthread_create(&tid[i], 0, thread, NULL);
	}
	for (i = 0; i < N; i++)
		pthread_join(tid[i], 0);
}
*/
import "C"

import (
	"os"
	"os/signal"
	"syscall"
	"testing"
	"time"
)

func test3250(t *testing.T) {
	t.Skip("skipped, see golang.org/issue/5885")
	const (
		thres = 1
		sig   = syscall.SIGCHLD
	)
	type result struct {
		n   int
		sig os.Signal
	}
	var (
		sigCh     = make(chan os.Signal, 10)
		waitStart = make(chan struct{})
		waitDone  = make(chan result)
	)

	signal.Notify(sigCh, sig)

	go func() {
		n := 0
		alarm := time.After(time.Second * 3)
		for {
			select {
			case <-waitStart:
				waitStart = nil
			case v := <-sigCh:
				n++
				if v != sig || n > thres {
					waitDone <- result{n, v}
					return
				}
			case <-alarm:
				waitDone <- result{n, sig}
				return
			}
		}
	}()

	waitStart <- struct{}{}
	C.testSendSIG()
	r := <-waitDone
	if r.sig != sig {
		t.Fatalf("received signal %v, but want %v", r.sig, sig)
	}
	t.Logf("got %d signals\n", r.n)
	if r.n <= thres {
		t.Fatalf("expected more than %d", thres)
	}
}
