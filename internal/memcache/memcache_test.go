// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memcache

import (
	"context"
	"os"
	"testing"
	"time"
)

func getClient(t *testing.T) *Client {
	t.Helper()

	addr := os.Getenv("GOLANG_REDIS_ADDR")
	if addr == "" {
		t.Skip("skipping because GOLANG_REDIS_ADDR is unset")
	}

	return New(addr)
}

func TestCacheMiss(t *testing.T) {
	c := getClient(t)
	ctx := context.Background()

	if _, err := c.Get(ctx, "doesnotexist"); err != ErrCacheMiss {
		t.Errorf("got %v; want ErrCacheMiss", err)
	}
}

func TestExpiry(t *testing.T) {
	c := getClient(t).WithCodec(Gob)
	ctx := context.Background()

	key := "testexpiry"

	firstTime := time.Now()
	err := c.Set(ctx, &Item{
		Key:        key,
		Object:     firstTime,
		Expiration: 3500 * time.Millisecond, // NOTE: check that non-rounded expiries work.
	})
	if err != nil {
		t.Fatalf("Set: %v", err)
	}

	var newTime time.Time
	if err := c.Get(ctx, key, &newTime); err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !firstTime.Equal(newTime) {
		t.Errorf("Get: got value %v, want %v", newTime, firstTime)
	}

	time.Sleep(4 * time.Second)

	if err := c.Get(ctx, key, &newTime); err != ErrCacheMiss {
		t.Errorf("Get: got %v, want ErrCacheMiss", err)
	}
}

func TestShortExpiry(t *testing.T) {
	c := getClient(t).WithCodec(Gob)
	ctx := context.Background()

	key := "testshortexpiry"

	err := c.Set(ctx, &Item{
		Key:        key,
		Value:      []byte("ok"),
		Expiration: time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Set: %v", err)
	}

	if err := c.Get(ctx, key, nil); err != ErrCacheMiss {
		t.Errorf("GetBytes: got %v, want ErrCacheMiss", err)
	}
}
