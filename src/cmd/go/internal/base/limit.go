// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"internal/godebug"
	"runtime"
	"strconv"
	"sync"
)

var NetLimitGodebug = godebug.New("#cmdgonetlimit")

// NetLimit returns the limit on concurrent network operations
// configured by GODEBUG=cmdgonetlimit, if any.
//
// A limit of 0 (indicated by 0, true) means that network operations should not
// be allowed.
func NetLimit() (int, bool) {
	netLimitOnce.Do(func() {
		s := NetLimitGodebug.Value()
		if s == "" {
			return
		}

		n, err := strconv.Atoi(s)
		if err != nil {
			Fatalf("invalid %s: %v", NetLimitGodebug.Name(), err)
		}
		if n < 0 {
			// Treat negative values as unlimited.
			return
		}
		netLimitSem = make(chan struct{}, n)
	})

	return cap(netLimitSem), netLimitSem != nil
}

// AcquireNet acquires a semaphore token for a network operation.
func AcquireNet() (release func(), err error) {
	hasToken := false
	if n, ok := NetLimit(); ok {
		if n == 0 {
			return nil, fmt.Errorf("network disabled by %v=%v", NetLimitGodebug.Name(), NetLimitGodebug.Value())
		}
		netLimitSem <- struct{}{}
		hasToken = true
	}

	checker := new(netTokenChecker)
	runtime.SetFinalizer(checker, (*netTokenChecker).panicUnreleased)

	return func() {
		if checker.released {
			panic("internal error: net token released twice")
		}
		checker.released = true
		if hasToken {
			<-netLimitSem
		}
		runtime.SetFinalizer(checker, nil)
	}, nil
}

var (
	netLimitOnce sync.Once
	netLimitSem  chan struct{}
)

type netTokenChecker struct {
	released bool
	// We want to use a finalizer to check that all acquired tokens are returned,
	// so we arbitrarily pad the tokens with a string to defeat the runtime's
	// “tiny allocator”.
	unusedAvoidTinyAllocator string
}

func (c *netTokenChecker) panicUnreleased() {
	panic("internal error: net token acquired but not released")
}
