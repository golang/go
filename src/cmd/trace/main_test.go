// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

func TestListenAddr(t *testing.T) {
	tests := []struct {
		name     string
		addr     string
		wantAddr string
		wantErr  bool
	}{
		{
			name:     "empty host",
			addr:     ":8080",
			wantAddr: "localhost:8080",
		},
		{
			name:     "with host",
			addr:     "localhost:8080",
			wantAddr: "localhost:8080",
		},
		{
			name:     "with IP",
			addr:     "127.0.0.1:8080",
			wantAddr: "127.0.0.1:8080",
		},
		{
			name:     "unspecified host",
			addr:     "0.0.0.0:8080",
			wantAddr: "0.0.0.0:8080",
		},
		{
			name:    "host only",
			addr:    "127.0.0.1",
			wantErr: true,
		},
		{
			name:    "port only",
			addr:    "8080",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := listenAddr(tt.addr)
			if tt.wantErr && err == nil {
				t.Errorf("listenAddr(%q) got nil err want non-nil", tt.addr)
			} else if !tt.wantErr && err != nil {
				t.Errorf("listenAddr(%q) got err %v want nil", tt.addr, err)
			} else if got != tt.wantAddr {
				t.Errorf("listenAddr(%q) = %q, want %q", tt.addr, got, tt.wantAddr)
			}
		})
	}
}

func TestAddrURL(t *testing.T) {
	tests := []struct {
		name           string
		addr           string
		wantURL        string
		wantSimplified bool
	}{
		{
			name:           "empty host",
			addr:           ":8080",
			wantURL:        "http://localhost:8080",
			wantSimplified: true,
		},
		{
			name:           "with host",
			addr:           "localhost:8080",
			wantURL:        "http://localhost:8080",
			wantSimplified: false,
		},
		{
			name:           "with ip",
			addr:           "10.10.10.10:8080",
			wantURL:        "http://10.10.10.10:8080",
			wantSimplified: false,
		},
		{
			name:           "unspecified ipv4",
			addr:           "0.0.0.0:8080",
			wantURL:        "http://localhost:8080",
			wantSimplified: true,
		},
		{
			name:           "unspecified ipv6",
			addr:           "[::]:8080",
			wantURL:        "http://localhost:8080",
			wantSimplified: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotURL, gotSimplified, err := addrURL(tt.addr)
			if err != nil {
				t.Fatalf("addrURL(%q) got err %v want nil", tt.addr, err)
			}
			if gotURL != tt.wantURL {
				t.Errorf("addrURL(%q) = %q, want %q", tt.addr, gotURL, tt.wantURL)
			}
			if gotSimplified != tt.wantSimplified {
				t.Errorf("addrURL(%q) simplified = %v, want %v", tt.addr, gotSimplified, tt.wantSimplified)
			}
		})
	}
}
