// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(js && wasm)
// +build !js !wasm

package http

import (
	"context"
	"net"
	"time"
)

func defaultTransportDialContext() func(context.Context, string, string) (net.Conn, error) {
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}
	return dialer.DialContext
}
