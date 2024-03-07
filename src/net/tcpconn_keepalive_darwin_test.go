// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package net

import (
	"syscall"
	"testing"
	"time"
)

func getCurrentKeepAliveSettings(fd int) (cfg KeepAliveConfig, err error) {
	tcpKeepAlive, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE)
	if err != nil {
		return
	}
	tcpKeepAliveIdle, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE)
	if err != nil {
		return
	}
	tcpKeepAliveInterval, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, sysTCP_KEEPINTVL)
	if err != nil {
		return
	}
	tcpKeepAliveCount, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, sysTCP_KEEPCNT)
	if err != nil {
		return
	}
	cfg = KeepAliveConfig{
		Enable:   tcpKeepAlive != 0,
		Idle:     time.Duration(tcpKeepAliveIdle) * time.Second,
		Interval: time.Duration(tcpKeepAliveInterval) * time.Second,
		Count:    tcpKeepAliveCount,
	}
	return
}

func verifyKeepAliveSettings(t *testing.T, fd int, oldCfg, cfg KeepAliveConfig) {
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

	tcpKeepAliveIdle, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveIdle)*time.Second != cfg.Idle {
		t.Fatalf("TCP_KEEPIDLE: got %ds; want %v", tcpKeepAliveIdle, cfg.Idle)
	}

	tcpKeepAliveInterval, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, sysTCP_KEEPINTVL)
	if err != nil {
		t.Fatal(err)
	}
	if time.Duration(tcpKeepAliveInterval)*time.Second != cfg.Interval {
		t.Fatalf("TCP_KEEPINTVL: got %ds; want %v", tcpKeepAliveInterval, cfg.Interval)
	}

	tcpKeepAliveCount, err := syscall.GetsockoptInt(fd, syscall.IPPROTO_TCP, sysTCP_KEEPCNT)
	if err != nil {
		t.Fatal(err)
	}
	if tcpKeepAliveCount != cfg.Count {
		t.Fatalf("TCP_KEEPCNT: got %d; want %d", tcpKeepAliveCount, cfg.Count)
	}
}
