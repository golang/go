// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/36114
 * Buggy version: 6d4d3c52ae7c3f910bfc7552a2a673a8338e5b9f
 * fix commit-id: a44fcd3d27c06aaa60d8d1cbce169f0d982e74b1
 * Flaky: 100/100
 */
package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Moby36114", Moby36114)
}

type serviceVM_moby36114 struct {
	sync.Mutex
}

func (svm *serviceVM_moby36114) hotAddVHDsAtStart() {
	svm.Lock()
	defer svm.Unlock()
	svm.hotRemoveVHDsAtStart()
}

func (svm *serviceVM_moby36114) hotRemoveVHDsAtStart() {
	svm.Lock()
	defer svm.Unlock()
}

func Moby36114() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			s := &serviceVM_moby36114{}
			go s.hotAddVHDsAtStart()
		}()
	}
}
