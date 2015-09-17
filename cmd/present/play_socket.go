// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine,!appenginevm

package main

import (
	"net/http"
	"net/url"
	"runtime"

	"golang.org/x/tools/playground/socket"
	"golang.org/x/tools/present"
)

func initPlayground(basepath string, origin *url.URL) {
	if present.PlayEnabled {
		if *nativeClient {
			socket.RunScripts = false
			socket.Environ = func() []string {
				if runtime.GOARCH == "amd64" {
					return environ("GOOS=nacl", "GOARCH=amd64p32")
				}
				return environ("GOOS=nacl")
			}
		}
		playScript(basepath, "SocketTransport")
		http.Handle("/socket", socket.NewHandler(origin))
	}
}

func playable(c present.Code) bool {
	return present.PlayEnabled && c.Play
}
