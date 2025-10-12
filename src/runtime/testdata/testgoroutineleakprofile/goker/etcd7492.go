// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: etcd
 * Issue or PR  : https://github.com/etcd-io/etcd/pull/7492
 * Buggy version: 51939650057d602bb5ab090633138fffe36854dc
 * fix commit-id: 1b1fabef8ffec606909f01c3983300fff539f214
 * Flaky: 40/100
 */
package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Etcd7492", Etcd7492)
}

type TokenProvider_etcd7492 interface {
	assign()
	enable()
	disable()
}

type simpleTokenTTLKeeper_etcd7492 struct {
	tokens           map[string]time.Time
	addSimpleTokenCh chan struct{}
	stopCh           chan chan struct{}
	deleteTokenFunc  func(string)
}

type authStore_etcd7492 struct {
	tokenProvider TokenProvider_etcd7492
}

func (as *authStore_etcd7492) Authenticate() {
	as.tokenProvider.assign()
}

func NewSimpleTokenTTLKeeper_etcd7492(deletefunc func(string)) *simpleTokenTTLKeeper_etcd7492 {
	stk := &simpleTokenTTLKeeper_etcd7492{
		tokens:           make(map[string]time.Time),
		addSimpleTokenCh: make(chan struct{}, 1),
		stopCh:           make(chan chan struct{}),
		deleteTokenFunc:  deletefunc,
	}
	go stk.run() // G1
	return stk
}

func (tm *simpleTokenTTLKeeper_etcd7492) run() {
	tokenTicker := time.NewTicker(time.Nanosecond)
	defer tokenTicker.Stop()
	for {
		select {
		case <-tm.addSimpleTokenCh:
			runtime.Gosched()
			/// Make tm.tokens not empty is enough
			tm.tokens["1"] = time.Now()
		case <-tokenTicker.C:
			runtime.Gosched()
			for t, _ := range tm.tokens {
				tm.deleteTokenFunc(t)
				delete(tm.tokens, t)
			}
		case waitCh := <-tm.stopCh:
			waitCh <- struct{}{}
			return
		}
	}
}

func (tm *simpleTokenTTLKeeper_etcd7492) addSimpleToken() {
	tm.addSimpleTokenCh <- struct{}{}
	runtime.Gosched()
}

func (tm *simpleTokenTTLKeeper_etcd7492) stop() {
	waitCh := make(chan struct{})
	tm.stopCh <- waitCh
	<-waitCh
	close(tm.stopCh)
}

type tokenSimple_etcd7492 struct {
	simpleTokenKeeper *simpleTokenTTLKeeper_etcd7492
	simpleTokensMu    sync.RWMutex
}

func (t *tokenSimple_etcd7492) assign() {
	t.assignSimpleTokenToUser()
}

func (t *tokenSimple_etcd7492) assignSimpleTokenToUser() {
	t.simpleTokensMu.Lock()
	runtime.Gosched()
	t.simpleTokenKeeper.addSimpleToken()
	t.simpleTokensMu.Unlock()
}
func newDeleterFunc(t *tokenSimple_etcd7492) func(string) {
	return func(tk string) {
		t.simpleTokensMu.Lock()
		defer t.simpleTokensMu.Unlock()
	}
}

func (t *tokenSimple_etcd7492) enable() {
	t.simpleTokenKeeper = NewSimpleTokenTTLKeeper_etcd7492(newDeleterFunc(t))
}

func (t *tokenSimple_etcd7492) disable() {
	if t.simpleTokenKeeper != nil {
		t.simpleTokenKeeper.stop()
		t.simpleTokenKeeper = nil
	}
	t.simpleTokensMu.Lock()
	t.simpleTokensMu.Unlock()
}

func newTokenProviderSimple_etcd7492() *tokenSimple_etcd7492 {
	return &tokenSimple_etcd7492{}
}

func setupAuthStore_etcd7492() (store *authStore_etcd7492, teardownfunc func()) {
	as := &authStore_etcd7492{
		tokenProvider: newTokenProviderSimple_etcd7492(),
	}
	as.tokenProvider.enable()
	tearDown := func() {
		as.tokenProvider.disable()
	}
	return as, tearDown
}

func Etcd7492() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 100; i++ {
		go func() {
			as, tearDown := setupAuthStore_etcd7492()
			defer tearDown()
			var wg sync.WaitGroup
			wg.Add(3)
			for i := 0; i < 3; i++ {
				go func() { // G2
					as.Authenticate()
					defer wg.Done()
				}()
			}
			wg.Wait()
		}()
	}
}
