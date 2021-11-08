// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <sys/mman.h>
#include <pthread.h>
#include <unistd.h>

void ctor(void) __attribute__((constructor));
static void* thread(void*);

void
ctor(void)
{
	// occupy memory where Go runtime would normally map heap
	mmap((void*)0x00c000000000, 64<<10, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0);

	// allocate 4K every 10us
	pthread_t t;
	pthread_create(&t, 0, thread, 0);
}

static void*
thread(void *p)
{
	for(;;) {
		usleep(10000);
		mmap(0, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
	}
	return 0;
}
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

func main() {
	start := time.Now()

	// ensure that we can function normally
	var v [][]byte
	for i := 0; i < 1000; i++ {
		time.Sleep(10 * time.Microsecond)
		v = append(v, make([]byte, 64<<10))
	}

	fmt.Printf("ok\t%s\t%s\n", filepath.Base(os.Args[0]), time.Since(start).Round(time.Millisecond))
}
