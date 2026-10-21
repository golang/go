// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"
)

// defaultDebounce is how long the proxy waits after the last edit before
// it recompiles in the background.
const defaultDebounce = 300 * time.Millisecond

// stopGrace is how long a stopping proxy waits for gopls to exit before it
// kills it.
const stopGrace = 3 * time.Second

// drainLimit is how long a proxy whose client closed its output waits for
// the answers to the requests still in flight.
const drainLimit = 30 * time.Second

// Options configures a Proxy.
type Options struct {
	// Gopls is the command that runs gopls over stdio, program first.
	Gopls []string

	// Env is the environment of gopls; nil inherits the proxy's.
	Env []string

	// Stderr receives the stderr of gopls and the proxy's own log lines;
	// nil means os.Stderr.
	Stderr io.Writer

	// Compiler lowers sources and reports compiler diagnostics.
	Compiler Compiler

	// Debounce is the delay between the last edit and the background
	// recompile. Zero means 300 ms; a negative value turns background
	// recompiles off, leaving the recompile before each request and on
	// open and save.
	Debounce time.Duration

	// Hints serves compiler hints for inlayHint and hover requests, and
	// HintsJSON produces the hints JSON of one package. Both nil means the
	// proxy serves only what gopls reports. See HintsFunc.
	Hints     HintsFunc
	HintsJSON HintsJSONFunc

	// CodeActions adds the code actions of the compiler autofix candidates
	// to the answer of a codeAction request. See CodeActionsFunc.
	CodeActions CodeActionsFunc
}

// document is an open document of the client.
type document struct {
	uri     string
	path    string
	version int
	text    *text
}

// unit is one open package directory: the dependency closure it was last
// compiled with, and what that compile produced.
type unit struct {
	dir   string
	dirs  map[string]bool
	paths map[string]bool
	diags []Diagnostic
	dirty bool

	// diagVersions holds the version of each open document at the time diags
	// was computed, so diagnostics are never shown against a newer text.
	diagVersions map[string]int

	// gen counts the changes of the unit's inputs and results, so work
	// computed from an older state is never stored as current.
	gen int

	hints   json.RawMessage
	hintsOK bool
}

// virtDoc is a lowered document open in gopls.
type virtDoc struct {
	lw      *lowered
	version int
}

// serverDiags are the diagnostics gopls last published for one document,
// with the version of the document they describe.
type serverDiags struct {
	version int
	hasVer  bool
	raw     json.RawMessage
}

// pending is a request forwarded to gopls whose response is yet to come.
// It keeps the tables the request was mapped with, so the response maps
// back through the same lowerings even when a recompile has replaced them.
// A codeAction request also keeps the version of the document it was
// asked about, so a crossed edit drops the whole codeAction answer
// instead of mixing the old range with new diagnostics.
type pending struct {
	method string
	path   string
	lw     *lowered
	tb     *tables
	params json.RawMessage
	// fixVersion is the version of the document at request time, and
	// fixHas reports whether the document was open then. Only codeAction
	// requests set them.
	fixVersion int
	fixHas     bool
}

// Proxy sits between an LSP client and gopls. Positions and URIs of loGo/s
// sources travel to gopls as the lowered Go documents the compiler produces
// for them, and travel back as positions in the sources.
type Proxy struct {
	opts   Options
	client *writer
	server *writer
	tables atomic.Pointer[tables]

	// work serializes every sequence that compiles, updates the lowered
	// documents of gopls, or forwards a request to it, so a request always
	// meets the lowerings it was mapped with. It is never taken by the
	// loop that reads gopls.
	work sync.Mutex

	// pub serializes diagnostics publication.
	pub sync.Mutex

	// tasks counts the background work: the debounce timer callback and
	// the hint merges.
	tasks sync.WaitGroup

	mu       sync.Mutex
	docs     map[string]*document
	units    map[string]*unit
	files    map[string]*lowered
	virt     map[string]*virtDoc
	pending  map[string]pending
	srvDiags map[string]serverDiags
	sent     map[string]string
	timer    *time.Timer
	stopped  bool
	shutdown bool
	settled  chan struct{}
}

// newProxy returns a proxy writing client messages to out.
func newProxy(out io.Writer, opts Options) *Proxy {
	if opts.Stderr == nil {
		opts.Stderr = os.Stderr
	}
	opts.Debounce = cmp.Or(opts.Debounce, defaultDebounce)
	p := &Proxy{
		opts:     opts,
		client:   &writer{w: out},
		docs:     make(map[string]*document),
		units:    make(map[string]*unit),
		files:    make(map[string]*lowered),
		virt:     make(map[string]*virtDoc),
		pending:  make(map[string]pending),
		srvDiags: make(map[string]serverDiags),
		sent:     make(map[string]string),
		settled:  make(chan struct{}, 1),
	}
	p.tables.Store(&tables{})
	return p
}

// logf writes one line to the proxy's log.
func (p *Proxy) logf(format string, args ...any) {
	fmt.Fprintf(p.opts.Stderr, "logos lsp: "+format+"\n", args...)
}

// server is the running gopls process. done closes when the loop that
// reads its output has ended, and err then holds why. reap alone waits for
// cmd, and kills it when it ignores the closed input; exec.CommandContext
// also kills it when the context given to start ends.
type server struct {
	cmd  *exec.Cmd
	done chan struct{}
	err  error
}

// reap waits for gopls to end after its input is closed, kills it when it
// outlives stopGrace, and collects the process. Whatever it returns, gopls
// is waited for. ctx is the context given to start.
//
// It returns how gopls ended only when a clean exit or a signal sent on
// purpose does not explain it: reap's own kill, or the kill that the
// cancellation of ctx delivers. A kill that finds gopls already gone is no
// error. A kill that fails is returned together with the wait error; reap
// still waits for the process then, because returning without Wait would
// leave the child unreaped, and a cancelled ctx may still end it.
func (s *server) reap(ctx context.Context) error {
	var errs []error
	killed := false
	select {
	case <-s.done:
	case <-time.After(stopGrace):
		if err := s.cmd.Process.Kill(); err == nil {
			killed = true
		} else if !errors.Is(err, os.ErrProcessDone) {
			errs = append(errs, fmt.Errorf("killing gopls: %w", err))
		}
		<-s.done
	}
	if err := s.cmd.Wait(); err != nil {
		_, exited := errors.AsType[*exec.ExitError](err)
		signalled := (killed || ctx.Err() != nil) && (exited || errors.Is(err, ctx.Err()))
		if !signalled {
			errs = append(errs, fmt.Errorf("waiting for gopls: %w", err))
		}
	}
	return errors.Join(errs...)
}

// Serve runs a proxy that reads client messages from in and writes
// responses and notifications to out, until the client sends exit, closes
// in, or gopls dies. When the client closes in, the proxy first answers the
// requests still in flight. It starts gopls itself, and stops it, and its
// own background work, before it returns.
func Serve(ctx context.Context, in io.Reader, out io.Writer, opts Options) error {
	if len(opts.Gopls) == 0 || opts.Compiler == nil {
		return errors.New("logos lsp: no gopls command or compiler")
	}
	workCtx, cancelWork := context.WithCancel(ctx)
	defer cancelWork()
	p := newProxy(out, opts)
	srv, err := p.start(ctx, workCtx)
	if err != nil {
		return err
	}
	clientDone := make(chan struct{})
	var clientErr error
	go func() {
		clientErr = p.clientLoop(workCtx, newReader(in))
		close(clientDone)
	}()
	select {
	case <-clientDone:
		err = clientErr
		if err == nil {
			p.drain(srv.done)
		}
	case <-srv.done:
		err = cmp.Or(srv.err, errors.New("gopls exited unexpectedly"))
	}
	p.stop(ctx, srv, cancelWork)
	closeInput(in, clientDone)
	return err
}

// closeInput closes the client's input, when it can be closed, to end the
// loop that reads it, and waits for that loop, but not for long: a reader
// that closing cannot interrupt, a terminal, may keep it for the life of
// the process.
func closeInput(in io.Reader, loopDone <-chan struct{}) {
	if c, ok := in.(io.Closer); ok {
		c.Close()
	}
	select {
	case <-loopDone:
	case <-time.After(stopGrace):
	}
}

// start launches gopls and the loop that forwards its messages. Cancelling
// ctx kills gopls; workCtx is the context of the proxy's own work.
func (p *Proxy) start(ctx, workCtx context.Context) (*server, error) {
	cmd := exec.CommandContext(ctx, p.opts.Gopls[0], p.opts.Gopls[1:]...)
	cmd.Env = p.opts.Env
	cmd.Stderr = p.opts.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting gopls: %w", err)
	}
	p.server = &writer{w: stdin}
	srv := &server{cmd: cmd, done: make(chan struct{})}
	go func() {
		srv.err = p.serverLoop(workCtx, newReader(stdout))
		close(srv.done)
	}()
	return srv, nil
}

// drain waits until no request is in flight, gopls is gone, or the limit
// passes.
func (p *Proxy) drain(gopls <-chan struct{}) {
	limit := time.After(drainLimit)
	for {
		p.mu.Lock()
		n := len(p.pending)
		p.mu.Unlock()
		if n == 0 {
			return
		}
		select {
		case <-p.settled:
		case <-gopls:
			return
		case <-limit:
			return
		}
	}
}

// stop ends the background work, closes gopls's input, and waits for it,
// killing it when it ignores the closed input, then releases the compiler.
// No background work starts after stop marks the proxy stopped, and a
// compile already running is cancelled with cancelWork and awaited. An end
// of gopls that the shutdown does not explain (see server.reap) goes to the
// log; it does not change what Serve returns. ctx is the context given to
// start.
func (p *Proxy) stop(ctx context.Context, srv *server, cancelWork context.CancelFunc) {
	p.mu.Lock()
	p.stopped = true
	if p.timer != nil && p.timer.Stop() {
		p.tasks.Done()
	}
	p.mu.Unlock()
	cancelWork()
	p.tasks.Wait()
	if c, ok := p.server.w.(io.Closer); ok {
		c.Close()
	}
	if err := srv.reap(ctx); err != nil {
		p.logf("gopls: %v", err)
	}
	p.work.Lock()
	defer p.work.Unlock()
	if c, ok := p.opts.Compiler.(io.Closer); ok {
		c.Close()
	}
}

// clientLoop handles client messages until the client exits. It returns
// nil at the end of input, and an error for an exit without a shutdown.
func (p *Proxy) clientLoop(ctx context.Context, rd *reader) error {
	for {
		m, err := rd.read()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if m.Method == "exit" {
			return p.exit(m)
		}
		p.fromClient(ctx, m)
	}
}

// exit forwards the exit notification and reports whether the client shut
// the server down first.
func (p *Proxy) exit(m *message) error {
	p.send(p.server, m)
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.shutdown {
		return errors.New("logos lsp: exit without shutdown")
	}
	return nil
}

// serverLoop handles gopls messages until its output ends.
func (p *Proxy) serverLoop(ctx context.Context, rd *reader) error {
	for {
		m, err := rd.read()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		p.fromServer(ctx, m)
	}
}

// send writes m to w and logs a failure. A failed write means the peer is
// gone; the read loops notice that on their own.
func (p *Proxy) send(w *writer, m *message) {
	if err := w.write(m); err != nil {
		p.logf("write: %v", err)
	}
}

// addPending records the request id as awaiting its response.
func (p *Proxy) addPending(id json.RawMessage, pend pending) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pending[string(id)] = pend
}

// finish removes the request id from the pending set, and wakes a drain.
func (p *Proxy) finish(id json.RawMessage) {
	p.mu.Lock()
	delete(p.pending, string(id))
	p.mu.Unlock()
	select {
	case p.settled <- struct{}{}:
	default:
	}
}
