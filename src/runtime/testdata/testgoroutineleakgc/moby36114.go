/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/36114
 * Buggy version: 6d4d3c52ae7c3f910bfc7552a2a673a8338e5b9f
 * fix commit-id: a44fcd3d27c06aaa60d8d1cbce169f0d982e74b1
 * Flaky: 100/100
 * Description:
 *   This is a double lock bug. The the lock for the
 * struct svm has already been locked when calling
 * svm.hotRemoveVHDsAtStart()
 */
package main

import (
	"runtime"
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
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()

	for i := 0; i < 100; i++ {
		go func() {
			s := &serviceVM_moby36114{}
			// deadlocks: x > 0
			go s.hotAddVHDsAtStart()
		}()
	}
}
