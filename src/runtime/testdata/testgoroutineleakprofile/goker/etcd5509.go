// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"io"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Etcd5509", Etcd5509)
}

var ErrConnClosed_etcd5509 error

type Client_etcd5509 struct {
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
}

func (c *Client_etcd5509) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancel == nil {
		return
	}
	c.cancel()
	c.cancel = nil
	c.mu.Unlock()
	c.mu.Lock()
}

type remoteClient_etcd5509 struct {
	client *Client_etcd5509
	mu     sync.Mutex
}

func (r *remoteClient_etcd5509) acquire(ctx context.Context) error {
	for {
		r.client.mu.RLock()
		closed := r.client.cancel == nil
		r.mu.Lock()
		r.mu.Unlock()
		if closed {
			return ErrConnClosed_etcd5509 // Missing RUnlock before return
		}
		r.client.mu.RUnlock()
	}
}

type kv_etcd5509 struct {
	rc *remoteClient_etcd5509
}

func (kv *kv_etcd5509) Get(ctx context.Context) error {
	return kv.Do(ctx)
}

func (kv *kv_etcd5509) Do(ctx context.Context) error {
	for {
		err := kv.do(ctx)
		if err == nil {
			return nil
		}
		return err
	}
}

func (kv *kv_etcd5509) do(ctx context.Context) error {
	err := kv.getRemote(ctx)
	return err
}

func (kv *kv_etcd5509) getRemote(ctx context.Context) error {
	return kv.rc.acquire(ctx)
}

type KV interface {
	Get(ctx context.Context) error
	Do(ctx context.Context) error
}

func NewKV_etcd5509(c *Client_etcd5509) KV {
	return &kv_etcd5509{rc: &remoteClient_etcd5509{
		client: c,
	}}
}

func Etcd5509() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() {
		ctx, _ := context.WithCancel(context.TODO())
		cli := &Client_etcd5509{
			ctx: ctx,
		}
		kv := NewKV_etcd5509(cli)
		donec := make(chan struct{})
		go func() {
			defer close(donec)
			err := kv.Get(context.TODO())
			if err != nil && err != ErrConnClosed_etcd5509 {
				io.Discard.Write([]byte("Expect ErrConnClosed"))
			}
		}()

		runtime.Gosched()
		cli.Close()

		<-donec
	}()
}
