// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build solaris

package net

import (
	"syscall"
	"testing"
	"time"
)

var testConfigs = []KeepAliveConfig{
	{
		Enable:   true,
		Idle:     2 * time.Second,
		Interval: -1,
		Count:    -1,
	},
	{
		Enable:   true,
		Idle:     0,
		Interval: -1,
		Count:    -1,
	},
	{
		Enable:   true,
		Idle:     -1,
		Interval: -1,
		Count:    -1,
	},
}

func getCurrentKeepAliveSettings(fd fdType) (cfg KeepAliveConfig, err error) {
	tcpKeepAlive, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE)
	if err != nil {
		return
	}
	tcpKeepAliveIdle, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_THRESHOLD)
	if err != nil {
		return
	}
	cfg = KeepAliveConfig{
		Enable:   tcpKeepAlive != 0,
		Idle:     time.Duration(tcpKeepAliveIdle) * time.Millisecond,
		Interval: -1,
		Count:    -1,
	}
	return
}

func verifyKeepAliveSettings(t *testing.T, fd fdType, oldCfg, cfg KeepAliveConfig) {
	if cfg.Idle == 0 {
		cfg.Idle = defaultTCPKeepAliveIdle
	}
	if cfg.Interval == 0 {
		cfg.Interval = defaultTCPKeepAliveInterval
	}
	if cfg.Count == 0 {
		cfg.Count = defaultTCPKeepAliveCount
	}
	if cfg.Idle == -1 {
		cfg.Idle = oldCfg.Idle
	}
	if cfg.Interval == -1 {
		cfg.Interval = oldCfg.Interval
	}
	if cfg.Count == -1 {
		cfg.Count = oldCfg.Count
	}

	tcpKeepAlive, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE)
	if err != nil {
		t.Fatal(err)
	}
	if (tcpKeepAlive != 0) != cfg.Enable {
		t.Fatalf("SO_KEEPALIVE: got %t; want %t", tcpKeepAlive != 0, cfg.Enable)
	}

	tcpKeepAliveIdle, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_THRESHOLD)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveIdle)*time.Millisecond != cfg.Idle {
		t.Fatalf("TCP_KEEPIDLE: got %dms; want %v", tcpKeepAliveIdle, cfg.Idle)
	}
}
