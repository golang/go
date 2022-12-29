// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gocommand

import (
	"strconv"
	"testing"
)

func TestParseGoVersionOutput(t *testing.T) {
	tests := []struct {
		args string
		want string
	}{
		{"go version go1.12 linux/amd64", "go1.12"},
		{"go version go1.18.1 darwin/amd64", "go1.18.1"},
		{"go version go1.19.rc1 windows/arm64", "go1.19.rc1"},
		{"go version devel d5de62df152baf4de6e9fe81933319b86fd95ae4 linux/386", "devel d5de62df152baf4de6e9fe81933319b86fd95ae4"},
		{"go version devel go1.20-1f068f0dc7 Tue Oct 18 20:58:37 2022 +0000 darwin/amd64", "devel go1.20-1f068f0dc7"},
		{"v1.19.1 foo/bar", ""},
	}
	for i, tt := range tests {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if got := ParseGoVersionOutput(tt.args); got != tt.want {
				t.Errorf("parseGoVersionOutput() = %v, want %v", got, tt.want)
			}
		})
	}
}
