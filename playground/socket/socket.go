// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine

// Package socket implements an WebSocket-based playground backend.
// Clients connect to a websocket handler and send run/kill commands, and
// the server sends the output and exit status of the running processes.
// Multiple clients running multiple processes may be served concurrently.
// The wire format is JSON and is described by the Message type.
//
// This will not run on App Engine as WebSockets are not supported there.
package socket // import "golang.org/x/tools/playground/socket"

import (
	"bytes"
	"encoding/json"
	"errors"
	"go/parser"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"golang.org/x/net/websocket"
)

// RunScripts specifies whether the socket handler should execute shell scripts
// (snippets that start with a shebang).
var RunScripts = true

// Environ provides an environment when a binary, such as the go tool, is
// invoked.
var Environ func() []string = os.Environ

const (
	// The maximum number of messages to send per session (avoid flooding).
	msgLimit = 1000

	// Batch messages sent in this interval and send as a single message.
	msgDelay = 10 * time.Millisecond
)

// Message is the wire format for the websocket connection to the browser.
// It is used for both sending output messages and receiving commands, as
// distinguished by the Kind field.
type Message struct {
	Id      string // client-provided unique id for the process
	Kind    string // in: "run", "kill" out: "stdout", "stderr", "end"
	Body    string
	Options *Options `json:",omitempty"`
}

// Options specify additional message options.
type Options struct {
	Race bool // use -race flag when building code (for "run" only)
}

// NewHandler returns a websocket server which checks the origin of requests.
func NewHandler(origin *url.URL) websocket.Server {
	return websocket.Server{
		Config:    websocket.Config{Origin: origin},
		Handshake: handshake,
		Handler:   websocket.Handler(socketHandler),
	}
}

// handshake checks the origin of a request during the websocket handshake.
func handshake(c *websocket.Config, req *http.Request) error {
	o, err := websocket.Origin(c, req)
	if err != nil {
		log.Println("bad websocket origin:", err)
		return websocket.ErrBadWebSocketOrigin
	}
	_, port, err := net.SplitHostPort(c.Origin.Host)
	if err != nil {
		log.Println("bad websocket origin:", err)
		return websocket.ErrBadWebSocketOrigin
	}
	ok := c.Origin.Scheme == o.Scheme && (c.Origin.Host == o.Host || c.Origin.Host == net.JoinHostPort(o.Host, port))
	if !ok {
		log.Println("bad websocket origin:", o)
		return websocket.ErrBadWebSocketOrigin
	}
	log.Println("accepting connection from:", req.RemoteAddr)
	return nil
}

// socketHandler handles the websocket connection for a given present session.
// It handles transcoding Messages to and from JSON format, and starting
// and killing processes.
func socketHandler(c *websocket.Conn) {
	in, out := make(chan *Message), make(chan *Message)
	errc := make(chan error, 1)

	// Decode messages from client and send to the in channel.
	go func() {
		dec := json.NewDecoder(c)
		for {
			var m Message
			if err := dec.Decode(&m); err != nil {
				errc <- err
				return
			}
			in <- &m
		}
	}()

	// Receive messages from the out channel and encode to the client.
	go func() {
		enc := json.NewEncoder(c)
		for m := range out {
			if err := enc.Encode(m); err != nil {
				errc <- err
				return
			}
		}
	}()
	defer close(out)

	// Start and kill processes and handle errors.
	proc := make(map[string]*process)
	for {
		select {
		case m := <-in:
			switch m.Kind {
			case "run":
				log.Println("running snippet from:", c.Request().RemoteAddr)
				proc[m.Id].Kill()
				proc[m.Id] = startProcess(m.Id, m.Body, out, m.Options)
			case "kill":
				proc[m.Id].Kill()
			}
		case err := <-errc:
			if err != io.EOF {
				// A encode or decode has failed; bail.
				log.Println(err)
			}
			// Shut down any running processes.
			for _, p := range proc {
				p.Kill()
			}
			return
		}
	}
}

// process represents a running process.
type process struct {
	out  chan<- *Message
	done chan struct{} // closed when wait completes
	run  *exec.Cmd
	bin  string
}

// startProcess builds and runs the given program, sending its output
// and end event as Messages on the provided channel.
func startProcess(id, body string, dest chan<- *Message, opt *Options) *process {
	var (
		done = make(chan struct{})
		out  = make(chan *Message)
		p    = &process{out: out, done: done}
	)
	go func() {
		defer close(done)
		for m := range buffer(limiter(out, p)) {
			m.Id = id
			dest <- m
		}
	}()
	var err error
	if path, args := shebang(body); path != "" {
		if RunScripts {
			err = p.startProcess(path, args, body)
		} else {
			err = errors.New("script execution is not allowed")
		}
	} else {
		err = p.start(body, opt)
	}
	if err != nil {
		p.end(err)
		return nil
	}
	go func() {
		p.end(p.run.Wait())
	}()
	return p
}

// end sends an "end" message to the client, containing the process id and the
// given error value. It also removes the binary, if present.
func (p *process) end(err error) {
	if p.bin != "" {
		defer os.Remove(p.bin)
	}
	m := &Message{Kind: "end"}
	if err != nil {
		m.Body = err.Error()
	}
	p.out <- m
	close(p.out)
}

// A killer provides a mechanism to terminate a process.
// The Kill method returns only once the process has exited.
type killer interface {
	Kill()
}

// limiter returns a channel that wraps the given channel.
// It receives Messages from the given channel and sends them to the returned
// channel until it passes msgLimit messages, at which point it will kill the
// process and pass only the "end" message.
// When the given channel is closed, or when the "end" message is received,
// it closes the returned channel.
func limiter(in <-chan *Message, p killer) <-chan *Message {
	out := make(chan *Message)
	go func() {
		defer close(out)
		n := 0
		for m := range in {
			switch {
			case n < msgLimit || m.Kind == "end":
				out <- m
				if m.Kind == "end" {
					return
				}
			case n == msgLimit:
				// Kill in a goroutine as Kill will not return
				// until the process' output has been
				// processed, and we're doing that in this loop.
				go p.Kill()
			default:
				continue // don't increment
			}
			n++
		}
	}()
	return out
}

// buffer returns a channel that wraps the given channel. It receives messages
// from the given channel and sends them to the returned channel.
// Message bodies are gathered over the period msgDelay and coalesced into a
// single Message before they are passed on. Messages of the same kind are
// coalesced; when a message of a different kind is received, any buffered
// messages are flushed. When the given channel is closed, buffer flushes the
// remaining buffered messages and closes the returned channel.
func buffer(in <-chan *Message) <-chan *Message {
	out := make(chan *Message)
	go func() {
		defer close(out)
		var (
			t     = time.NewTimer(msgDelay)
			tc    <-chan time.Time
			buf   []byte
			kind  string
			flush = func() {
				if len(buf) == 0 {
					return
				}
				out <- &Message{Kind: kind, Body: safeString(buf)}
				buf = buf[:0] // recycle buffer
				kind = ""
			}
		)
		for {
			select {
			case m, ok := <-in:
				if !ok {
					flush()
					return
				}
				if m.Kind == "end" {
					flush()
					out <- m
					return
				}
				if kind != m.Kind {
					flush()
					kind = m.Kind
					if tc == nil {
						tc = t.C
						t.Reset(msgDelay)
					}
				}
				buf = append(buf, m.Body...)
			case <-tc:
				flush()
				tc = nil
			}
		}
	}()
	return out
}

// Kill stops the process if it is running and waits for it to exit.
func (p *process) Kill() {
	if p == nil || p.run == nil {
		return
	}
	p.run.Process.Kill()
	<-p.done // block until process exits
}

// shebang looks for a shebang ('#!') at the beginning of the passed string.
// If found, it returns the path and args after the shebang.
// args includes the command as args[0].
func shebang(body string) (path string, args []string) {
	body = strings.TrimSpace(body)
	if !strings.HasPrefix(body, "#!") {
		return "", nil
	}
	if i := strings.Index(body, "\n"); i >= 0 {
		body = body[:i]
	}
	fs := strings.Fields(body[2:])
	return fs[0], fs
}

// startProcess starts a given program given its path and passing the given body
// to the command standard input.
func (p *process) startProcess(path string, args []string, body string) error {
	cmd := &exec.Cmd{
		Path:   path,
		Args:   args,
		Stdin:  strings.NewReader(body),
		Stdout: &messageWriter{kind: "stdout", out: p.out},
		Stderr: &messageWriter{kind: "stderr", out: p.out},
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	p.run = cmd
	return nil
}

// start builds and starts the given program, sending its output to p.out,
// and stores the running *exec.Cmd in the run field.
func (p *process) start(body string, opt *Options) error {
	// We "go build" and then exec the binary so that the
	// resultant *exec.Cmd is a handle to the user's program
	// (rather than the go tool process).
	// This makes Kill work.

	bin := filepath.Join(tmpdir, "compile"+strconv.Itoa(<-uniq))
	src := bin + ".go"
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}

	// write body to x.go
	defer os.Remove(src)
	err := ioutil.WriteFile(src, []byte(body), 0666)
	if err != nil {
		return err
	}

	// build x.go, creating x
	p.bin = bin // to be removed by p.end
	dir, file := filepath.Split(src)
	args := []string{"go", "build", "-tags", "OMIT"}
	if opt != nil && opt.Race {
		p.out <- &Message{
			Kind: "stderr",
			Body: "Running with race detector.\n",
		}
		args = append(args, "-race")
	}
	args = append(args, "-o", bin, file)
	cmd := p.cmd(dir, args...)
	cmd.Stdout = cmd.Stderr // send compiler output to stderr
	if err := cmd.Run(); err != nil {
		return err
	}

	// run x
	if isNacl() {
		cmd, err = p.naclCmd(bin)
		if err != nil {
			return err
		}
	} else {
		cmd = p.cmd("", bin)
	}
	if opt != nil && opt.Race {
		cmd.Env = append(cmd.Env, "GOMAXPROCS=2")
	}
	if err := cmd.Start(); err != nil {
		// If we failed to exec, that might be because they built
		// a non-main package instead of an executable.
		// Check and report that.
		if name, err := packageName(body); err == nil && name != "main" {
			return errors.New(`executable programs must use "package main"`)
		}
		return err
	}
	p.run = cmd
	return nil
}

// cmd builds an *exec.Cmd that writes its standard output and error to the
// process' output channel.
func (p *process) cmd(dir string, args ...string) *exec.Cmd {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = dir
	cmd.Env = Environ()
	cmd.Stdout = &messageWriter{kind: "stdout", out: p.out}
	cmd.Stderr = &messageWriter{kind: "stderr", out: p.out}
	return cmd
}

func isNacl() bool {
	for _, v := range append(Environ(), os.Environ()...) {
		if v == "GOOS=nacl" {
			return true
		}
	}
	return false
}

// naclCmd returns an *exec.Cmd that executes bin under native client.
func (p *process) naclCmd(bin string) (*exec.Cmd, error) {
	pwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	var args []string
	env := []string{
		"NACLENV_GOOS=" + runtime.GOOS,
		"NACLENV_GOROOT=/go",
		"NACLENV_NACLPWD=" + strings.Replace(pwd, runtime.GOROOT(), "/go", 1),
	}
	switch runtime.GOARCH {
	case "amd64":
		env = append(env, "NACLENV_GOARCH=amd64p32")
		args = []string{"sel_ldr_x86_64"}
	case "386":
		env = append(env, "NACLENV_GOARCH=386")
		args = []string{"sel_ldr_x86_32"}
	case "arm":
		env = append(env, "NACLENV_GOARCH=arm")
		selLdr, err := exec.LookPath("sel_ldr_arm")
		if err != nil {
			return nil, err
		}
		args = []string{"nacl_helper_bootstrap_arm", selLdr, "--reserved_at_zero=0xXXXXXXXXXXXXXXXX"}
	default:
		return nil, errors.New("native client does not support GOARCH=" + runtime.GOARCH)
	}

	cmd := p.cmd("", append(args, "-l", "/dev/null", "-S", "-e", bin)...)
	cmd.Env = append(cmd.Env, env...)

	return cmd, nil
}

func packageName(body string) (string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), "prog.go",
		strings.NewReader(body), parser.PackageClauseOnly)
	if err != nil {
		return "", err
	}
	return f.Name.String(), nil
}

// messageWriter is an io.Writer that converts all writes to Message sends on
// the out channel with the specified id and kind.
type messageWriter struct {
	kind string
	out  chan<- *Message
}

func (w *messageWriter) Write(b []byte) (n int, err error) {
	w.out <- &Message{Kind: w.kind, Body: safeString(b)}
	return len(b), nil
}

// safeString returns b as a valid UTF-8 string.
func safeString(b []byte) string {
	if utf8.Valid(b) {
		return string(b)
	}
	var buf bytes.Buffer
	for len(b) > 0 {
		r, size := utf8.DecodeRune(b)
		b = b[size:]
		buf.WriteRune(r)
	}
	return buf.String()
}

var tmpdir string

func init() {
	// find real path to temporary directory
	var err error
	tmpdir, err = filepath.EvalSymlinks(os.TempDir())
	if err != nil {
		log.Fatal(err)
	}
}

var uniq = make(chan int) // a source of numbers for naming temporary files

func init() {
	go func() {
		for i := 0; ; i++ {
			uniq <- i
		}
	}()
}
