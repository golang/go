// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build golangorg

// Package memcache provides a minimally compatible interface for
// google.golang.org/appengine/memcache
// and stores the data in Redis (e.g., via Cloud Memorystore).
package memcache

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"time"

	"github.com/gomodule/redigo/redis"
)

var ErrCacheMiss = errors.New("memcache: cache miss")

func New(addr string) *Client {
	const maxConns = 20

	pool := redis.NewPool(func() (redis.Conn, error) {
		return redis.Dial("tcp", addr)
	}, maxConns)

	return &Client{
		pool: pool,
	}
}

type Client struct {
	pool *redis.Pool
}

type CodecClient struct {
	client *Client
	codec  Codec
}

type Item struct {
	Key        string
	Value      []byte
	Object     interface{}   // Used with Codec.
	Expiration time.Duration // Read-only.
}

func (c *Client) WithCodec(codec Codec) *CodecClient {
	return &CodecClient{
		c, codec,
	}
}

func (c *Client) Delete(ctx context.Context, key string) error {
	conn, err := c.pool.GetContext(ctx)
	if err != nil {
		return err
	}
	defer conn.Close()

	_, err = conn.Do("DEL", key)
	return err
}

func (c *CodecClient) Delete(ctx context.Context, key string) error {
	return c.client.Delete(ctx, key)
}

func (c *Client) Set(ctx context.Context, item *Item) error {
	if item.Value == nil {
		return errors.New("nil item value")
	}
	return c.set(ctx, item.Key, item.Value, item.Expiration)
}

func (c *CodecClient) Set(ctx context.Context, item *Item) error {
	if item.Object == nil {
		return errors.New("nil object value")
	}
	b, err := c.codec.Marshal(item.Object)
	if err != nil {
		return err
	}
	return c.client.set(ctx, item.Key, b, item.Expiration)
}

func (c *Client) set(ctx context.Context, key string, value []byte, expiration time.Duration) error {
	conn, err := c.pool.GetContext(ctx)
	if err != nil {
		return err
	}
	defer conn.Close()

	if expiration == 0 {
		_, err := conn.Do("SET", key, value)
		return err
	}

	// NOTE(cbro): redis does not support expiry in units more granular than a second.
	exp := int64(expiration.Seconds())
	if exp == 0 {
		// Redis doesn't allow a zero expiration, delete the key instead.
		_, err := conn.Do("DEL", key)
		return err
	}

	_, err = conn.Do("SETEX", key, exp, value)
	return err
}

// Get gets the item.
func (c *Client) Get(ctx context.Context, key string) ([]byte, error) {
	conn, err := c.pool.GetContext(ctx)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	b, err := redis.Bytes(conn.Do("GET", key))
	if err == redis.ErrNil {
		err = ErrCacheMiss
	}
	return b, err
}

func (c *CodecClient) Get(ctx context.Context, key string, v interface{}) error {
	b, err := c.client.Get(ctx, key)
	if err != nil {
		return err
	}
	return c.codec.Unmarshal(b, v)
}

var (
	Gob  = Codec{gobMarshal, gobUnmarshal}
	JSON = Codec{json.Marshal, json.Unmarshal}
)

type Codec struct {
	Marshal   func(interface{}) ([]byte, error)
	Unmarshal func([]byte, interface{}) error
}

func gobMarshal(v interface{}) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gobUnmarshal(data []byte, v interface{}) error {
	return gob.NewDecoder(bytes.NewBuffer(data)).Decode(v)
}
