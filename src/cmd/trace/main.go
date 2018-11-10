// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"cmd/internal/browser"
	"flag"
	"fmt"
	"html/template"
	"internal/trace"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
)

const usageMessage = "" +
	`Usage of 'go tool trace':
Given a trace file produced by 'go test':
	go test -trace=trace.out pkg

Open a web browser displaying trace:
	go tool trace [flags] [pkg.test] trace.out

Generate a pprof-like profile from the trace:
    go tool trace -pprof=TYPE [pkg.test] trace.out

[pkg.test] argument is required for traces produced by Go 1.6 and below.
Go 1.7 does not require the binary argument.

Supported profile types are:
    - net: network blocking profile
    - sync: synchronization blocking profile
    - syscall: syscall blocking profile
    - sched: scheduler latency profile

Flags:
	-http=addr: HTTP service address (e.g., ':6060')
	-pprof=type: print a pprof-like profile instead

Note that while the various profiles available when launching
'go tool trace' work on every browser, the trace viewer itself
(the 'view trace' page) comes from the Chrome/Chromium project
and is only actively tested on that browser.
`

var (
	httpFlag  = flag.String("http", "localhost:0", "HTTP service address (e.g., ':6060')")
	pprofFlag = flag.String("pprof", "", "print a pprof-like profile instead")

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

	// Go 1.7 traces embed symbol info and does not require the binary.
	// But we optionally accept binary as first arg for Go 1.5 traces.
	switch flag.NArg() {
	case 1:
		traceFile = flag.Arg(0)
	case 2:
		programBinary = flag.Arg(0)
		traceFile = flag.Arg(1)
	default:
		flag.Usage()
	}

	var pprofFunc func(io.Writer) error
	switch *pprofFlag {
	case "net":
		pprofFunc = pprofIO
	case "sync":
		pprofFunc = pprofBlock
	case "syscall":
		pprofFunc = pprofSyscall
	case "sched":
		pprofFunc = pprofSched
	}
	if pprofFunc != nil {
		if err := pprofFunc(os.Stdout); err != nil {
			dief("failed to generate pprof: %v\n", err)
		}
		os.Exit(0)
	}
	if *pprofFlag != "" {
		dief("unknown pprof type %s\n", *pprofFlag)
	}

	ln, err := net.Listen("tcp", *httpFlag)
	if err != nil {
		dief("failed to create server socket: %v\n", err)
	}

	log.Printf("Parsing trace...")
	events, err := parseEvents()
	if err != nil {
		dief("%v\n", err)
	}

	log.Printf("Serializing trace...")
	params := &traceParams{
		events:  events,
		endTime: int64(1<<63 - 1),
	}
	data, err := generateTrace(params)
	if err != nil {
		dief("%v\n", err)
	}

	log.Printf("Splitting trace...")
	ranges = splitTrace(data)

	log.Printf("Opening browser")
	if !browser.Open("http://" + ln.Addr().String()) {
		fmt.Fprintf(os.Stderr, "Trace viewer is listening on http://%s\n", ln.Addr().String())
	}

	// Start http server.
	http.HandleFunc("/", httpMain)
	err = http.Serve(ln, nil)
	dief("failed to start http server: %v\n", err)
}

var ranges []Range

var loader struct {
	once   sync.Once
	events []*trace.Event
	err    error
}

func parseEvents() ([]*trace.Event, error) {
	loader.once.Do(func() {
		tracef, err := os.Open(traceFile)
		if err != nil {
			loader.err = fmt.Errorf("failed to open trace file: %v", err)
			return
		}
		defer tracef.Close()

		// Parse and symbolize.
		events, err := trace.Parse(bufio.NewReader(tracef), programBinary)
		if err != nil {
			loader.err = fmt.Errorf("failed to parse trace: %v", err)
			return
		}
		loader.events = events
	})
	return loader.events, loader.err
}

// httpMain serves the starting page.
func httpMain(w http.ResponseWriter, r *http.Request) {
	if err := templMain.Execute(w, ranges); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

var templMain = template.Must(template.New("").Parse(`
<html>
<body>
{{if $}}
	{{range $e := $}}
		<a href="/trace?start={{$e.Start}}&end={{$e.End}}">View trace ({{$e.Name}})</a><br>
	{{end}}
	<br>
{{else}}
	<a href="/trace">View trace</a><br>
{{end}}
<a href="/goroutines">Goroutine analysis</a><br>
<a href="/io">Network blocking profile</a><br>
<a href="/block">Synchronization blocking profile</a><br>
<a href="/syscall">Syscall blocking profile</a><br>
<a href="/sched">Scheduler latency profile</a><br>
</body>
</html>
`))

func dief(msg string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, msg, args...)
	os.Exit(1)
}
