// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine

package main

import (
	"flag"
	"fmt"
	"go/build"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strings"

	"code.google.com/p/go.tools/playground/socket"
	"code.google.com/p/go.tools/present"
)

const basePkg = "code.google.com/p/go.tools/cmd/present"

var (
	basePath     string
	nativeClient bool
)

func main() {
	httpListen := flag.String("http", "127.0.0.1:3999", "host:port to listen on")
	flag.StringVar(&basePath, "base", "", "base path for slide template and static resources")
	flag.BoolVar(&present.PlayEnabled, "play", true, "enable playground (permit execution of arbitrary user code)")
	flag.BoolVar(&nativeClient, "nacl", false, "use Native Client environment playground (prevents non-Go code execution)")
	flag.Parse()

	if basePath == "" {
		p, err := build.Default.Import(basePkg, "", build.FindOnly)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Couldn't find gopresent files: %v\n", err)
			fmt.Fprintf(os.Stderr, basePathMessage, basePkg)
			os.Exit(1)
		}
		basePath = p.Dir
	}

	if present.PlayEnabled {
		if nativeClient {
			socket.RunScripts = false
			socket.Environ = func() []string {
				if runtime.GOARCH == "amd64" {
					return environ("GOOS=nacl", "GOARCH=amd64p32")
				}
				return environ("GOOS=nacl")
			}
		}
		playScript(basePath, "SocketTransport")

		host, port, err := net.SplitHostPort(*httpListen)
		if err != nil {
			log.Fatal(err)
		}
		origin := &url.URL{Scheme: "http", Host: host + ":" + port}
		http.Handle("/socket", socket.NewHandler(origin))
	}
	http.Handle("/static/", http.FileServer(http.Dir(basePath)))

	if !strings.HasPrefix(*httpListen, "127.0.0.1") &&
		!strings.HasPrefix(*httpListen, "localhost") &&
		present.PlayEnabled && !nativeClient {
		log.Print(localhostWarning)
	}

	log.Printf("Open your web browser and visit http://%s/", *httpListen)
	log.Fatal(http.ListenAndServe(*httpListen, nil))
}

func playable(c present.Code) bool {
	return present.PlayEnabled && c.Play
}

func environ(vars ...string) []string {
	env := os.Environ()
	for _, r := range vars {
		k := strings.SplitAfter(r, "=")[0]
		var found bool
		for i, v := range env {
			if strings.HasPrefix(v, k) {
				env[i] = r
				found = true
			}
		}
		if !found {
			env = append(env, r)
		}
	}
	return env
}

const basePathMessage = `
By default, gopresent locates the slide template files and associated
static content by looking for a %q package
in your Go workspaces (GOPATH).

You may use the -base flag to specify an alternate location.
`

const localhostWarning = `
WARNING!  WARNING!  WARNING!

The present server appears to be listening on an address that is not localhost.
Anyone with access to this address and port will have access to this machine as
the user running present.

To avoid this message, listen on localhost or run with -play=false.

If you don't understand this message, hit Control-C to terminate this process.

WARNING!  WARNING!  WARNING!
`
