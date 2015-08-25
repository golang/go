// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Trace is a tool for viewing trace files.

Trace files can be generated with:
	- runtime/trace.Start
	- net/http/pprof package
	- go test -trace

Example usage:
Generate a trace file with 'go test':
	go test -trace trace.out pkg
View the trace in a web browser:
	go tool trace pkg.test trace.out
*/
package main

import (
	"bufio"
	"flag"
	"fmt"
	"internal/trace"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"sync"
)

const usageMessage = "" +
	`Usage of 'go tool trace':
Given a trace file produced by 'go test':
	go test -trace=trace.out pkg

Open a web browser displaying trace:
	go tool trace [flags] pkg.test trace.out

Flags:
	-http=addr: HTTP service address (e.g., ':6060')
`

var (
	httpFlag = flag.String("http", "localhost:0", "HTTP service address (e.g., ':6060')")

	// The binary file name, left here for serveSVGProfile.
	programBinary string
	traceFile     string
)

func main() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, usageMessage)
		os.Exit(2)
	}
	flag.Parse()

	// Usage information when no arguments.
	if flag.NArg() != 2 {
		flag.Usage()
	}
	programBinary = flag.Arg(0)
	traceFile = flag.Arg(1)

	ln, err := net.Listen("tcp", *httpFlag)
	if err != nil {
		dief("failed to create server socket: %v\n", err)
	}
	// Open browser.
	if !startBrowser("http://" + ln.Addr().String()) {
		fmt.Fprintf(os.Stderr, "Trace viewer is listening on http://%s\n", ln.Addr().String())
	}

	// Parse and symbolize trace asynchronously while browser opens.
	go parseEvents()

	// Start http server.
	http.HandleFunc("/", httpMain)
	err = http.Serve(ln, nil)
	dief("failed to start http server: %v\n", err)
}

var loader struct {
	once   sync.Once
	events []*trace.Event
	err    error
}

func parseEvents() ([]*trace.Event, error) {
	loader.once.Do(func() {
		tracef, err := os.Open(flag.Arg(1))
		if err != nil {
			loader.err = fmt.Errorf("failed to open trace file: %v", err)
			return
		}
		defer tracef.Close()

		// Parse and symbolize.
		events, err := trace.Parse(bufio.NewReader(tracef))
		if err != nil {
			loader.err = fmt.Errorf("failed to parse trace: %v", err)
			return
		}
		err = trace.Symbolize(events, programBinary)
		if err != nil {
			loader.err = fmt.Errorf("failed to symbolize trace: %v", err)
			return
		}
		loader.events = events
	})
	return loader.events, loader.err
}

// httpMain serves the starting page.
func httpMain(w http.ResponseWriter, r *http.Request) {
	w.Write(templMain)
}

var templMain = []byte(`
<html>
<body>
<a href="/trace">View trace</a><br>
<a href="/goroutines">Goroutine analysis</a><br>
<a href="/io">Network blocking profile</a><br>
<a href="/block">Synchronization blocking profile</a><br>
<a href="/syscall">Syscall blocking profile</a><br>
<a href="/sched">Scheduler latency profile</a><br>
</body>
</html>
`)

// startBrowser tries to open the URL in a browser
// and reports whether it succeeds.
// Note: copied from x/tools/cmd/cover/html.go
func startBrowser(url string) bool {
	// try to start the browser
	var args []string
	switch runtime.GOOS {
	case "darwin":
		args = []string{"open"}
	case "windows":
		args = []string{"cmd", "/c", "start"}
	default:
		args = []string{"xdg-open"}
	}
	cmd := exec.Command(args[0], append(args[1:], url)...)
	return cmd.Start() == nil
}

func dief(msg string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, msg, args...)
	os.Exit(1)
}
