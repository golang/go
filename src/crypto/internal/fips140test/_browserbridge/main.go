// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// browserbridge runs the crypto/internal/fips140test WebAssembly module in a
// web browser, driven from the host. It relays the BoringSSL acvptool module
// wrapper protocol for ACVP algorithm testing, and runs the FIPS 140-3
// functional tests, against the browser's WebAssembly engine.
//
// The bridge runs as a long-lived server on the host:
//
//	browserbridge -serve -wasm fips140test.test -wasmexec wasm_exec.js
//
// It prints a URL to open (once) in the browser under test. The page loads the
// WebAssembly module and waits for sessions. Each session instantiates a fresh
// module instance, equivalent to one process execution, with the argv and
// environment provided by a client.
//
// The browser may run on a different machine: the page is served at -addr and
// works over plain HTTP, the URL contains an unguessable token. Clients
// connect over a Unix socket, as acvptool runs on the same host as the
// bridge.
//
// Invoked with no flags, browserbridge acts as a module wrapper for acvptool:
//
//	ACVP_WRAPPER=1 acvptool -json vectors.json -wrapper ./browserbridge
//
// It relays its standard input and output to the module running in the browser,
// forwards the relevant environment variables, and exits with the module's
// exit code. It connects to the server over a fixed-name Unix socket in the
// user cache directory.
//
// The -run flag instead executes the module with the given arguments, to run
// the functional tests in the browser:
//
//	browserbridge -run -- -test.run 'TestIntegrityCheck|TestFIPS140' -test.v
//
// Some functional tests re-exec the test binary (TestCASTPasses,
// TestCASTFailures, TestIntegrityCheckFailure), which can't be done on
// js/wasm. Those run on the host with GOBROWSERBRIDGE pointing at this binary;
// they exec it (browserbridge -run) in place of themselves, so the re-exec'd
// module instance runs in the browser and the host test checks the relayed
// output. See crypto/internal/fips140test.
//
// TestIntegrityCheckFailure additionally passes -corrupt, which makes the
// bridge serve, for that one session, a module whose go:fipsinfo checksum has
// been overwritten, so the in-browser integrity check fails as it would for a
// tampered binary.
package main

import (
	"bytes"
	"crypto/rand"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed index.html
var indexHTML []byte

// Clients and the server exchange length-prefixed frames over a Unix socket:
// a one-byte type, a uint32 big-endian payload length, and the payload.
//
// Client to server:
//
//	'H' hello, a JSON-encoded hello structure; must be the first frame
//	'I' standard input bytes for the module
//	'E' standard input EOF
//
// Server to client:
//
//	'O' standard output bytes from the module
//	'R' standard error bytes from the module
//	'X' module exit, a one-byte exit code; the connection closes after
const maxFrame = 1 << 26

type hello struct {
	Argv    []string          `json:"argv"`
	Env     map[string]string `json:"env"`
	Corrupt bool              `json:"corrupt,omitempty"`
}

func writeFrame(w io.Writer, typ byte, payload []byte) error {
	hdr := [5]byte{typ}
	binary.BigEndian.PutUint32(hdr[1:], uint32(len(payload)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func readFrame(r io.Reader) (byte, []byte, error) {
	var hdr [5]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return 0, nil, err
	}
	n := binary.BigEndian.Uint32(hdr[1:])
	if n > maxFrame {
		return 0, nil, fmt.Errorf("oversized frame: %d bytes", n)
	}
	payload := make([]byte, n)
	if _, err := io.ReadFull(r, payload); err != nil {
		return 0, nil, err
	}
	return hdr[0], payload, nil
}

func main() {
	log.SetPrefix("browserbridge: ")
	log.SetFlags(log.Ltime)
	serve := flag.Bool("serve", false, "run the bridge server")
	run := flag.Bool("run", false, "run the module with the given arguments")
	corrupt := flag.Bool("corrupt", false, "run a module with a corrupted integrity checksum (-run)")
	addr := flag.String("addr", "127.0.0.1:14140", "HTTP listen address (-serve)")
	wasm := flag.String("wasm", "fips140test.test", "path to the WebAssembly module (-serve)")
	wasmExec := flag.String("wasmexec", "wasm_exec.js", "path to wasm_exec.js from GOROOT/lib/wasm (-serve)")
	flag.Parse()

	switch {
	case *serve:
		runServer(*addr, *wasm, *wasmExec)
	default:
		argv := []string{"fips140test.test"}
		if *run {
			argv = append(argv, flag.Args()...)
		}
		os.Exit(runClient(argv, !*run, *corrupt))
	}
}

// socketPath returns the fixed path of the Unix socket clients use to reach
// the server.
func socketPath() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		log.Fatal(err)
	}
	return filepath.Join(dir, "browserbridge.sock")
}

// runClient connects to the server, asks it to run the module with the given
// argv, relays standard input (if pumpStdin is set) and output, and returns
// the module's exit code. If corrupt is set, the session runs a module with a
// corrupted integrity checksum.
func runClient(argv []string, pumpStdin, corrupt bool) int {
	sock := socketPath()
	conn, err := net.Dial("unix", sock)
	if err != nil {
		fmt.Fprintf(os.Stderr, "browserbridge: can't connect to %s (is browserbridge -serve running?): %v\n", sock, err)
		return 2
	}
	defer conn.Close()

	// Forward only the environment variables that affect the module, so the
	// session logs record exactly the configuration under test.
	env := make(map[string]string)
	for _, kv := range os.Environ() {
		k, v, _ := strings.Cut(kv, "=")
		switch k {
		case "GODEBUG", "GONOPAAPAI", "GOENTROPYSOURCEACVP":
			env[k] = v
		default:
			if strings.HasPrefix(k, "ACVP_") {
				env[k] = v
			}
		}
	}
	h, err := json.Marshal(hello{Argv: argv, Env: env, Corrupt: corrupt})
	if err != nil {
		log.Fatal(err)
	}
	if err := writeFrame(conn, 'H', h); err != nil {
		fmt.Fprintf(os.Stderr, "browserbridge: %v\n", err)
		return 2
	}

	if pumpStdin {
		go func() {
			buf := make([]byte, 32<<10)
			for {
				n, err := os.Stdin.Read(buf)
				if n > 0 {
					if err := writeFrame(conn, 'I', buf[:n]); err != nil {
						return
					}
				}
				if err != nil {
					writeFrame(conn, 'E', nil)
					return
				}
			}
		}()
	} else {
		writeFrame(conn, 'E', nil)
	}

	for {
		typ, payload, err := readFrame(conn)
		if err != nil {
			fmt.Fprintf(os.Stderr, "browserbridge: connection lost: %v\n", err)
			return 2
		}
		switch typ {
		case 'O':
			os.Stdout.Write(payload)
		case 'R':
			os.Stderr.Write(payload)
		case 'X':
			code := 0
			if len(payload) > 0 {
				code = int(payload[0])
			}
			return code
		}
	}
}

// corruptModule overwrites the go:fipsinfo Sum field in bin in place, and
// returns its previous value. The field is located by searching for the
// 16-byte magic that precedes it, which by construction appears nowhere else
// in a module binary (see crypto/internal/fips140/check).
func corruptModule(bin []byte) ([]byte, error) {
	// "\xff" + fipsMagic, assembled at run time to keep the sequence out of
	// this binary too.
	magic := append([]byte{0xff}, " Go fipsinfo \xff\x00"...)
	i := bytes.Index(bin, magic)
	if i < 0 {
		return nil, errors.New("go:fipsinfo magic not found (module not built with GOFIPS140?)")
	}
	rest := bin[i+len(magic):]
	if bytes.Contains(rest, magic) {
		return nil, errors.New("multiple go:fipsinfo magic occurrences")
	}
	if len(rest) < 32 {
		return nil, errors.New("module truncated after go:fipsinfo magic")
	}
	sum := bytes.Clone(rest[:32])
	copy(rest, bytes.Repeat([]byte("X"), 32))
	return sum, nil
}

// A session is one client connection, executed as one instantiation of the
// module in the browser.
type session struct {
	id      int
	argv    []string
	env     map[string]string
	corrupt bool // serve a module with a corrupted integrity checksum
	start   time.Time

	conn   net.Conn
	connMu sync.Mutex // serializes writeFrame calls

	// stdinR is read by the page's /stdin request; stdinW is fed from the
	// client's 'I' frames and closed on 'E' or client disconnection.
	stdinR *io.PipeReader
	stdinW *io.PipeWriter

	closeOnce sync.Once
}

func (s *session) writeFrame(typ byte, payload []byte) error {
	s.connMu.Lock()
	defer s.connMu.Unlock()
	return writeFrame(s.conn, typ, payload)
}

func (s *session) close() {
	s.closeOnce.Do(func() {
		s.stdinW.Close()
		s.conn.Close()
	})
}

type server struct {
	wasm     string
	wasmExec string
	token    string

	queue chan *session

	corruptOnce sync.Once
	corruptWasm []byte
	corruptErr  error

	mu       sync.Mutex
	sessions map[int]*session
	nextID   int
	agents   map[string]bool // logged User-Agent values
}

// corruptedModule returns the module bytes with the integrity checksum
// overwritten, reading and corrupting the module file once on first use.
func (srv *server) corruptedModule() ([]byte, error) {
	srv.corruptOnce.Do(func() {
		bin, err := os.ReadFile(srv.wasm)
		if err != nil {
			srv.corruptErr = err
			return
		}
		sum, err := corruptModule(bin)
		if err != nil {
			srv.corruptErr = err
			return
		}
		log.Printf("serving module with corrupted checksum (was %x)", sum)
		srv.corruptWasm = bin
	})
	return srv.corruptWasm, srv.corruptErr
}

func runServer(addr, wasm, wasmExec string) {
	for _, path := range []string{wasm, wasmExec} {
		if _, err := os.Stat(path); err != nil {
			log.Fatal(err)
		}
	}

	httpLn, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatal(err)
	}
	_, port, err := net.SplitHostPort(httpLn.Addr().String())
	if err != nil {
		log.Fatal(err)
	}

	// The browser may be on another machine (e.g. across a tailnet), but
	// clients run on this host, next to acvptool, over a Unix socket.
	sock := socketPath()
	if conn, err := net.Dial("unix", sock); err == nil {
		conn.Close()
		log.Fatalf("another browserbridge -serve is already running on %s", sock)
	}
	os.Remove(sock) // a previous run may have left a stale socket
	clientLn, err := net.Listen("unix", sock)
	if err != nil {
		log.Fatal(err)
	}

	srv := &server{
		wasm:     wasm,
		wasmExec: wasmExec,
		token:    rand.Text()[:5],
		queue:    make(chan *session, 64),
		sessions: make(map[int]*session),
		agents:   make(map[string]bool),
	}

	log.Printf("serving %s", wasm)
	urlHost := httpLn.Addr().String()
	if tcp, ok := httpLn.Addr().(*net.TCPAddr); ok && tcp.IP.IsUnspecified() {
		// Listening on all interfaces; the printed URL needs a reachable one.
		urlHost = net.JoinHostPort("<this machine's address>", port)
		if hostname, err := os.Hostname(); err == nil {
			urlHost = net.JoinHostPort(hostname, port)
		}
	}
	log.Printf("open http://%s/%s/ in the browser under test", urlHost, srv.token)
	log.Printf("clients connect to %s", sock)

	go func() {
		for {
			conn, err := clientLn.Accept()
			if err != nil {
				log.Fatal(err)
			}
			go srv.handleClient(conn)
		}
	}()

	log.Fatal(http.Serve(httpLn, srv.mux()))
}

func (srv *server) handleClient(conn net.Conn) {
	typ, payload, err := readFrame(conn)
	if err != nil || typ != 'H' {
		log.Printf("client %s: bad hello: %v", conn.RemoteAddr(), err)
		conn.Close()
		return
	}
	var h hello
	if err := json.Unmarshal(payload, &h); err != nil {
		log.Printf("client %s: bad hello: %v", conn.RemoteAddr(), err)
		conn.Close()
		return
	}
	if h.Env == nil {
		// The page indexes env unconditionally; don't deliver a JSON null.
		h.Env = map[string]string{}
	}

	stdinR, stdinW := io.Pipe()
	s := &session{
		argv:    h.Argv,
		env:     h.Env,
		corrupt: h.Corrupt,
		start:   time.Now(),
		conn:    conn,
		stdinR:  stdinR,
		stdinW:  stdinW,
	}
	srv.mu.Lock()
	srv.nextID++
	s.id = srv.nextID
	srv.sessions[s.id] = s
	srv.mu.Unlock()

	var envKeys []string
	for k, v := range h.Env {
		envKeys = append(envKeys, k+"="+v)
	}
	log.Printf("session %d: argv=%q env=%q", s.id, h.Argv, envKeys)
	srv.queue <- s

	for {
		typ, payload, err := readFrame(conn)
		if err != nil {
			break
		}
		switch typ {
		case 'I':
			// An error means the page side is gone; keep draining frames so
			// the client isn't stuck writing, it will notice on 'X' or close.
			stdinW.Write(payload)
		case 'E':
			stdinW.Close()
		}
	}
	s.close()
	srv.remove(s.id)
}

func (srv *server) remove(id int) {
	srv.mu.Lock()
	defer srv.mu.Unlock()
	delete(srv.sessions, id)
}

func (srv *server) session(r *http.Request) *session {
	id, err := strconv.Atoi(r.FormValue("s"))
	if err != nil {
		return nil
	}
	srv.mu.Lock()
	defer srv.mu.Unlock()
	return srv.sessions[id]
}

// checkOrigin rejects cross-origin browser requests. Same-origin requests
// either omit the Origin header or set it to this server's own host. The
// random token in the URL path is the primary protection; this is
// belt-and-suspenders against other websites driving the bridge.
func checkOrigin(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if o := r.Header.Get("Origin"); o != "" && o != "http://"+r.Host {
			http.Error(w, "cross-origin request rejected", http.StatusForbidden)
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		h(w, r)
	}
}

func (srv *server) mux() *http.ServeMux {
	mux := http.NewServeMux()
	t := "/" + srv.token
	mux.HandleFunc("GET "+t+"/{$}", checkOrigin(srv.handleIndex))
	mux.HandleFunc("GET "+t+"/wasm_exec.js", checkOrigin(func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, srv.wasmExec)
	}))
	mux.HandleFunc("GET "+t+"/module.wasm", checkOrigin(srv.handleModule))
	mux.HandleFunc("POST "+t+"/clientinfo", checkOrigin(srv.handleClientInfo))
	mux.HandleFunc("GET "+t+"/session", checkOrigin(srv.handleSession))
	mux.HandleFunc("GET "+t+"/stdin", checkOrigin(srv.handleStdin))
	mux.HandleFunc("POST "+t+"/io", checkOrigin(srv.handleIO))
	mux.HandleFunc("POST "+t+"/exit", checkOrigin(srv.handleExit))
	return mux
}

// handleModule serves the module. The page loads it once without a session,
// getting the real module; a corrupt session fetches it again with its id and
// gets a copy whose integrity checksum has been overwritten.
func (srv *server) handleModule(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/wasm")
	if s := srv.session(r); s != nil && s.corrupt {
		bin, err := srv.corruptedModule()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Write(bin)
		return
	}
	http.ServeFile(w, r, srv.wasm)
}

// handleClientInfo logs the browser environment reported by the page, as
// evidence for operational environment qualification. The page can identify
// the browser, version, and (on Chromium, in a secure context) the OS version
// and architecture; the host hardware model and CPU must come from the host
// running the browser.
func (srv *server) handleClientInfo(w http.ResponseWriter, r *http.Request) {
	payload, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 64<<10))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	log.Printf("client environment: %s", payload)
}

func (srv *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	srv.mu.Lock()
	logAgent := !srv.agents[r.UserAgent()]
	srv.agents[r.UserAgent()] = true
	srv.mu.Unlock()
	if logAgent {
		// Log the browser executing the module, as evidence in the
		// algorithm and functional testing logs.
		log.Printf("browser connected: %s", r.UserAgent())
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(indexHTML)
}

// handleSession long-polls for the next queued session. It returns 204 after
// a timeout, and the page polls again. It skips sessions whose client
// disconnected while queued, and requeues sessions it can't deliver (because
// the page went away mid-poll, e.g. on reload), so the client doesn't hang
// waiting for a session that was dequeued but never ran.
func (srv *server) handleSession(w http.ResponseWriter, r *http.Request) {
	timeout := time.After(50 * time.Second)
	for {
		var s *session
		select {
		case s = <-srv.queue:
		case <-timeout:
			w.WriteHeader(http.StatusNoContent)
			return
		case <-r.Context().Done():
			return
		}

		srv.mu.Lock()
		live := srv.sessions[s.id] == s
		srv.mu.Unlock()
		if !live {
			continue
		}

		if r.Context().Err() != nil {
			srv.requeue(s)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		err := json.NewEncoder(w).Encode(map[string]any{
			"id":      s.id,
			"argv":    s.argv,
			"env":     s.env,
			"corrupt": s.corrupt,
		})
		if err == nil {
			err = http.NewResponseController(w).Flush()
		}
		if err != nil {
			log.Printf("session %d: couldn't deliver to the page, requeueing: %v", s.id, err)
			srv.requeue(s)
		}
		return
	}
}

// requeue returns a session to the queue after a failed delivery to the page.
func (srv *server) requeue(s *session) {
	select {
	case srv.queue <- s:
	default:
		// The queue is full; drop the session so the client sees the
		// connection close instead of hanging.
		log.Printf("session %d: queue full, dropping", s.id)
		s.close()
		srv.remove(s.id)
	}
}

// handleStdin delivers one chunk of the module's standard input to the page,
// blocking until input is available. Each request returns as a complete
// response, and the page polls in a loop: a non-empty body is the next chunk
// of input, and an empty body with the Stdin-Eof header means the client has
// signaled EOF. A single long-lived streaming response would deadlock under
// Safari, which buffers the body of an in-progress response rather than
// surfacing it to the page incrementally, starving the module of input that
// has already arrived.
func (srv *server) handleStdin(w http.ResponseWriter, r *http.Request) {
	s := srv.session(r)
	if s == nil {
		http.Error(w, "no such session", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	buf := make([]byte, 32<<10)
	n, err := s.stdinR.Read(buf)
	if err != nil {
		// EOF or a closed pipe: no more input for this session.
		w.Header().Set("Stdin-Eof", "1")
		return
	}
	w.Write(buf[:n])
}

// handleIO relays module output bytes from the page to the client.
func (srv *server) handleIO(w http.ResponseWriter, r *http.Request) {
	s := srv.session(r)
	if s == nil {
		http.Error(w, "no such session", http.StatusNotFound)
		return
	}
	payload, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxFrame))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	typ := byte('O')
	if r.FormValue("fd") == "2" {
		typ = 'R'
	}
	if err := s.writeFrame(typ, payload); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
	}
}

func (srv *server) handleExit(w http.ResponseWriter, r *http.Request) {
	s := srv.session(r)
	if s == nil {
		http.Error(w, "no such session", http.StatusNotFound)
		return
	}
	var body struct {
		Code int `json:"code"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<10)).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	log.Printf("session %d: exit code %d (%s)", s.id, body.Code,
		time.Since(s.start).Round(10*time.Millisecond))
	err := s.writeFrame('X', []byte{byte(body.Code)})
	s.close()
	srv.remove(s.id)
	if err != nil && !errors.Is(err, net.ErrClosed) {
		http.Error(w, err.Error(), http.StatusBadGateway)
	}
}
