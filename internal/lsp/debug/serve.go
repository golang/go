// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"html/template"
	"io"
	stdlog "log"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"path"
	"path/filepath"
	"runtime"
	rpprof "runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/core"
	"golang.org/x/tools/internal/event/export"
	"golang.org/x/tools/internal/event/export/metric"
	"golang.org/x/tools/internal/event/export/ocagent"
	"golang.org/x/tools/internal/event/export/prometheus"
	"golang.org/x/tools/internal/event/keys"
	"golang.org/x/tools/internal/event/label"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug/log"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	errors "golang.org/x/xerrors"
)

type contextKeyType int

const (
	instanceKey contextKeyType = iota
	traceKey
)

// An Instance holds all debug information associated with a gopls instance.
type Instance struct {
	Logfile              string
	StartTime            time.Time
	ServerAddress        string
	DebugAddress         string
	ListenedDebugAddress string
	Workdir              string
	OCAgentConfig        string

	LogWriter io.Writer

	exporter event.Exporter

	ocagent    *ocagent.Exporter
	prometheus *prometheus.Exporter
	rpcs       *Rpcs
	traces     *traces
	State      *State
}

// State holds debugging information related to the server state.
type State struct {
	mu      sync.Mutex
	clients []*Client
	servers []*Server
}

// Caches returns the set of Cache objects currently being served.
func (st *State) Caches() []*cache.Cache {
	var caches []*cache.Cache
	seen := make(map[string]struct{})
	for _, client := range st.Clients() {
		cache, ok := client.Session.Cache().(*cache.Cache)
		if !ok {
			continue
		}
		if _, found := seen[cache.ID()]; found {
			continue
		}
		seen[cache.ID()] = struct{}{}
		caches = append(caches, cache)
	}
	return caches
}

// Cache returns the Cache that matches the supplied id.
func (st *State) Cache(id string) *cache.Cache {
	for _, c := range st.Caches() {
		if c.ID() == id {
			return c
		}
	}
	return nil
}

// Sessions returns the set of Session objects currently being served.
func (st *State) Sessions() []*cache.Session {
	var sessions []*cache.Session
	for _, client := range st.Clients() {
		sessions = append(sessions, client.Session)
	}
	return sessions
}

// Session returns the Session that matches the supplied id.
func (st *State) Session(id string) *cache.Session {
	for _, s := range st.Sessions() {
		if s.ID() == id {
			return s
		}
	}
	return nil
}

// Views returns the set of View objects currently being served.
func (st *State) Views() []*cache.View {
	var views []*cache.View
	for _, s := range st.Sessions() {
		for _, v := range s.Views() {
			if cv, ok := v.(*cache.View); ok {
				views = append(views, cv)
			}
		}
	}
	return views
}

// View returns the View that matches the supplied id.
func (st *State) View(id string) *cache.View {
	for _, v := range st.Views() {
		if v.ID() == id {
			return v
		}
	}
	return nil
}

// Clients returns the set of Clients currently being served.
func (st *State) Clients() []*Client {
	st.mu.Lock()
	defer st.mu.Unlock()
	clients := make([]*Client, len(st.clients))
	copy(clients, st.clients)
	return clients
}

// Client returns the Client matching the supplied id.
func (st *State) Client(id string) *Client {
	for _, c := range st.Clients() {
		if c.Session.ID() == id {
			return c
		}
	}
	return nil
}

// Servers returns the set of Servers the instance is currently connected to.
func (st *State) Servers() []*Server {
	st.mu.Lock()
	defer st.mu.Unlock()
	servers := make([]*Server, len(st.servers))
	copy(servers, st.servers)
	return servers
}

// A Client is an incoming connection from a remote client.
type Client struct {
	Session      *cache.Session
	DebugAddress string
	Logfile      string
	GoplsPath    string
	ServerID     string
	Service      protocol.Server
}

// A Server is an outgoing connection to a remote LSP server.
type Server struct {
	ID           string
	DebugAddress string
	Logfile      string
	GoplsPath    string
	ClientID     string
}

// AddClient adds a client to the set being served.
func (st *State) addClient(session *cache.Session) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.clients = append(st.clients, &Client{Session: session})
}

// DropClient removes a client from the set being served.
func (st *State) dropClient(session source.Session) {
	st.mu.Lock()
	defer st.mu.Unlock()
	for i, c := range st.clients {
		if c.Session == session {
			copy(st.clients[i:], st.clients[i+1:])
			st.clients[len(st.clients)-1] = nil
			st.clients = st.clients[:len(st.clients)-1]
			return
		}
	}
}

// AddServer adds a server to the set being queried. In practice, there should
// be at most one remote server.
func (st *State) addServer(server *Server) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.servers = append(st.servers, server)
}

// DropServer drops a server from the set being queried.
func (st *State) dropServer(id string) {
	st.mu.Lock()
	defer st.mu.Unlock()
	for i, s := range st.servers {
		if s.ID == id {
			copy(st.servers[i:], st.servers[i+1:])
			st.servers[len(st.servers)-1] = nil
			st.servers = st.servers[:len(st.servers)-1]
			return
		}
	}
}

// an http.ResponseWriter that filters writes
type filterResponse struct {
	w    http.ResponseWriter
	edit func([]byte) []byte
}

func (c filterResponse) Header() http.Header {
	return c.w.Header()
}

func (c filterResponse) Write(buf []byte) (int, error) {
	ans := c.edit(buf)
	return c.w.Write(ans)
}

func (c filterResponse) WriteHeader(n int) {
	c.w.WriteHeader(n)
}

// replace annoying nuls by spaces
func cmdline(w http.ResponseWriter, r *http.Request) {
	fake := filterResponse{
		w: w,
		edit: func(buf []byte) []byte {
			return bytes.ReplaceAll(buf, []byte{0}, []byte{' '})
		},
	}
	pprof.Cmdline(fake, r)
}

func (i *Instance) getCache(r *http.Request) interface{} {
	return i.State.Cache(path.Base(r.URL.Path))
}

func (i *Instance) getSession(r *http.Request) interface{} {
	return i.State.Session(path.Base(r.URL.Path))
}

func (i Instance) getClient(r *http.Request) interface{} {
	return i.State.Client(path.Base(r.URL.Path))
}

func (i Instance) getServer(r *http.Request) interface{} {
	i.State.mu.Lock()
	defer i.State.mu.Unlock()
	id := path.Base(r.URL.Path)
	for _, s := range i.State.servers {
		if s.ID == id {
			return s
		}
	}
	return nil
}

func (i Instance) getView(r *http.Request) interface{} {
	return i.State.View(path.Base(r.URL.Path))
}

func (i *Instance) getFile(r *http.Request) interface{} {
	identifier := path.Base(r.URL.Path)
	sid := path.Base(path.Dir(r.URL.Path))
	s := i.State.Session(sid)
	if s == nil {
		return nil
	}
	for _, o := range s.Overlays() {
		if o.FileIdentity().Hash == identifier {
			return o
		}
	}
	return nil
}

func (i *Instance) getInfo(r *http.Request) interface{} {
	buf := &bytes.Buffer{}
	i.PrintServerInfo(r.Context(), buf)
	return template.HTML(buf.String())
}

func (i *Instance) AddService(s protocol.Server, session *cache.Session) {
	for _, c := range i.State.clients {
		if c.Session == session {
			c.Service = s
			return
		}
	}
	stdlog.Printf("unable to find a Client to add the protocol.Server to")
}

func getMemory(r *http.Request) interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m
}

func init() {
	event.SetExporter(makeGlobalExporter(os.Stderr))
}

func GetInstance(ctx context.Context) *Instance {
	if ctx == nil {
		return nil
	}
	v := ctx.Value(instanceKey)
	if v == nil {
		return nil
	}
	return v.(*Instance)
}

// WithInstance creates debug instance ready for use using the supplied
// configuration and stores it in the returned context.
func WithInstance(ctx context.Context, workdir, agent string) context.Context {
	i := &Instance{
		StartTime:     time.Now(),
		Workdir:       workdir,
		OCAgentConfig: agent,
	}
	i.LogWriter = os.Stderr
	ocConfig := ocagent.Discover()
	//TODO: we should not need to adjust the discovered configuration
	ocConfig.Address = i.OCAgentConfig
	i.ocagent = ocagent.Connect(ocConfig)
	i.prometheus = prometheus.New()
	i.rpcs = &Rpcs{}
	i.traces = &traces{}
	i.State = &State{}
	i.exporter = makeInstanceExporter(i)
	return context.WithValue(ctx, instanceKey, i)
}

// SetLogFile sets the logfile for use with this instance.
func (i *Instance) SetLogFile(logfile string, isDaemon bool) (func(), error) {
	// TODO: probably a better solution for deferring closure to the caller would
	// be for the debug instance to itself be closed, but this fixes the
	// immediate bug of logs not being captured.
	closeLog := func() {}
	if logfile != "" {
		if logfile == "auto" {
			if isDaemon {
				logfile = filepath.Join(os.TempDir(), fmt.Sprintf("gopls-daemon-%d.log", os.Getpid()))
			} else {
				logfile = filepath.Join(os.TempDir(), fmt.Sprintf("gopls-%d.log", os.Getpid()))
			}
		}
		f, err := os.Create(logfile)
		if err != nil {
			return nil, errors.Errorf("unable to create log file: %w", err)
		}
		closeLog = func() {
			defer f.Close()
		}
		stdlog.SetOutput(io.MultiWriter(os.Stderr, f))
		i.LogWriter = f
	}
	i.Logfile = logfile
	return closeLog, nil
}

// Serve starts and runs a debug server in the background.
// It also logs the port the server starts on, to allow for :0 auto assigned
// ports.
func (i *Instance) Serve(ctx context.Context) error {
	stdlog.SetFlags(stdlog.Lshortfile)
	if i.DebugAddress == "" {
		return nil
	}
	listener, err := net.Listen("tcp", i.DebugAddress)
	if err != nil {
		return err
	}
	i.ListenedDebugAddress = listener.Addr().String()

	port := listener.Addr().(*net.TCPAddr).Port
	if strings.HasSuffix(i.DebugAddress, ":0") {
		stdlog.Printf("debug server listening at http://localhost:%d", port)
	}
	event.Log(ctx, "Debug serving", tag.Port.Of(port))
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/", render(MainTmpl, func(*http.Request) interface{} { return i }))
		mux.HandleFunc("/debug/", render(DebugTmpl, nil))
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		if i.prometheus != nil {
			mux.HandleFunc("/metrics/", i.prometheus.Serve)
		}
		if i.rpcs != nil {
			mux.HandleFunc("/rpc/", render(RPCTmpl, i.rpcs.getData))
		}
		if i.traces != nil {
			mux.HandleFunc("/trace/", render(TraceTmpl, i.traces.getData))
		}
		mux.HandleFunc("/cache/", render(CacheTmpl, i.getCache))
		mux.HandleFunc("/session/", render(SessionTmpl, i.getSession))
		mux.HandleFunc("/view/", render(ViewTmpl, i.getView))
		mux.HandleFunc("/client/", render(ClientTmpl, i.getClient))
		mux.HandleFunc("/server/", render(ServerTmpl, i.getServer))
		mux.HandleFunc("/file/", render(FileTmpl, i.getFile))
		mux.HandleFunc("/info", render(InfoTmpl, i.getInfo))
		mux.HandleFunc("/memory", render(MemoryTmpl, getMemory))
		if err := http.Serve(listener, mux); err != nil {
			event.Error(ctx, "Debug server failed", err)
			return
		}
		event.Log(ctx, "Debug server finished")
	}()
	return nil
}

// MonitorMemory starts recording memory statistics each second.
func (i *Instance) MonitorMemory(ctx context.Context) {
	tick := time.NewTicker(time.Second)
	nextThresholdGiB := uint64(1)
	go func() {
		for {
			<-tick.C
			var mem runtime.MemStats
			runtime.ReadMemStats(&mem)
			if mem.HeapAlloc < nextThresholdGiB*1<<30 {
				continue
			}
			if err := i.writeMemoryDebug(nextThresholdGiB, true); err != nil {
				event.Error(ctx, "writing memory debug info", err)
			}
			if err := i.writeMemoryDebug(nextThresholdGiB, false); err != nil {
				event.Error(ctx, "writing memory debug info", err)
			}
			event.Log(ctx, fmt.Sprintf("Wrote memory usage debug info to %v", os.TempDir()))
			nextThresholdGiB++
		}
	}()
}

func (i *Instance) writeMemoryDebug(threshold uint64, withNames bool) error {
	suffix := "withnames"
	if !withNames {
		suffix = "nonames"
	}

	filename := fmt.Sprintf("gopls.%d-%dGiB-%s.zip", os.Getpid(), threshold, suffix)
	zipf, err := os.OpenFile(filepath.Join(os.TempDir(), filename), os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	zipw := zip.NewWriter(zipf)

	f, err := zipw.Create("heap.pb.gz")
	if err != nil {
		return err
	}
	if err := rpprof.Lookup("heap").WriteTo(f, 0); err != nil {
		return err
	}

	f, err = zipw.Create("goroutines.txt")
	if err != nil {
		return err
	}
	if err := rpprof.Lookup("goroutine").WriteTo(f, 1); err != nil {
		return err
	}

	for _, cache := range i.State.Caches() {
		cf, err := zipw.Create(fmt.Sprintf("cache-%v.html", cache.ID()))
		if err != nil {
			return err
		}
		if _, err := cf.Write([]byte(cache.PackageStats(withNames))); err != nil {
			return err
		}
	}

	if err := zipw.Close(); err != nil {
		return err
	}
	return zipf.Close()
}

func makeGlobalExporter(stderr io.Writer) event.Exporter {
	p := export.Printer{}
	var pMu sync.Mutex
	return func(ctx context.Context, ev core.Event, lm label.Map) context.Context {
		i := GetInstance(ctx)

		if event.IsLog(ev) {
			// Don't log context cancellation errors.
			if err := keys.Err.Get(ev); errors.Is(err, context.Canceled) {
				return ctx
			}
			// Make sure any log messages without an instance go to stderr.
			if i == nil {
				pMu.Lock()
				p.WriteEvent(stderr, ev, lm)
				pMu.Unlock()
			}
			level := log.LabeledLevel(lm)
			// Exclude trace logs from LSP logs.
			if level < log.Trace {
				ctx = protocol.LogEvent(ctx, ev, lm, messageType(level))
			}
		}
		if i == nil {
			return ctx
		}
		return i.exporter(ctx, ev, lm)
	}
}

func messageType(l log.Level) protocol.MessageType {
	switch l {
	case log.Error:
		return protocol.Error
	case log.Warning:
		return protocol.Warning
	case log.Debug:
		return protocol.Log
	}
	return protocol.Info
}

func makeInstanceExporter(i *Instance) event.Exporter {
	exporter := func(ctx context.Context, ev core.Event, lm label.Map) context.Context {
		if i.ocagent != nil {
			ctx = i.ocagent.ProcessEvent(ctx, ev, lm)
		}
		if i.prometheus != nil {
			ctx = i.prometheus.ProcessEvent(ctx, ev, lm)
		}
		if i.rpcs != nil {
			ctx = i.rpcs.ProcessEvent(ctx, ev, lm)
		}
		if i.traces != nil {
			ctx = i.traces.ProcessEvent(ctx, ev, lm)
		}
		if event.IsLog(ev) {
			if s := cache.KeyCreateSession.Get(ev); s != nil {
				i.State.addClient(s)
			}
			if sid := tag.NewServer.Get(ev); sid != "" {
				i.State.addServer(&Server{
					ID:           sid,
					Logfile:      tag.Logfile.Get(ev),
					DebugAddress: tag.DebugAddress.Get(ev),
					GoplsPath:    tag.GoplsPath.Get(ev),
					ClientID:     tag.ClientID.Get(ev),
				})
			}
			if s := cache.KeyShutdownSession.Get(ev); s != nil {
				i.State.dropClient(s)
			}
			if sid := tag.EndServer.Get(ev); sid != "" {
				i.State.dropServer(sid)
			}
			if s := cache.KeyUpdateSession.Get(ev); s != nil {
				if c := i.State.Client(s.ID()); c != nil {
					c.DebugAddress = tag.DebugAddress.Get(ev)
					c.Logfile = tag.Logfile.Get(ev)
					c.ServerID = tag.ServerID.Get(ev)
					c.GoplsPath = tag.GoplsPath.Get(ev)
				}
			}
		}
		return ctx
	}
	// StdTrace must be above export.Spans below (by convention, export
	// middleware applies its wrapped exporter last).
	exporter = StdTrace(exporter)
	metrics := metric.Config{}
	registerMetrics(&metrics)
	exporter = metrics.Exporter(exporter)
	exporter = export.Spans(exporter)
	exporter = export.Labels(exporter)
	return exporter
}

type dataFunc func(*http.Request) interface{}

func render(tmpl *template.Template, fun dataFunc) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		var data interface{}
		if fun != nil {
			data = fun(r)
		}
		if err := tmpl.Execute(w, data); err != nil {
			event.Error(context.Background(), "", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func commas(s string) string {
	for i := len(s); i > 3; {
		i -= 3
		s = s[:i] + "," + s[i:]
	}
	return s
}

func fuint64(v uint64) string {
	return commas(strconv.FormatUint(v, 10))
}

func fuint32(v uint32) string {
	return commas(strconv.FormatUint(uint64(v), 10))
}

func fcontent(v []byte) string {
	return string(v)
}

var BaseTemplate = template.Must(template.New("").Parse(`
<html>
<head>
<title>{{template "title" .}}</title>
<style>
.profile-name{
	display:inline-block;
	width:6rem;
}
td.value {
  text-align: right;
}
ul.events {
	list-style-type: none;
}

</style>
{{block "head" .}}{{end}}
</head>
<body>
<a href="/">Main</a>
<a href="/info">Info</a>
<a href="/memory">Memory</a>
<a href="/metrics">Metrics</a>
<a href="/rpc">RPC</a>
<a href="/trace">Trace</a>
<hr>
<h1>{{template "title" .}}</h1>
{{block "body" .}}
Unknown page
{{end}}
</body>
</html>

{{define "cachelink"}}<a href="/cache/{{.}}">Cache {{.}}</a>{{end}}
{{define "clientlink"}}<a href="/client/{{.}}">Client {{.}}</a>{{end}}
{{define "serverlink"}}<a href="/server/{{.}}">Server {{.}}</a>{{end}}
{{define "sessionlink"}}<a href="/session/{{.}}">Session {{.}}</a>{{end}}
{{define "viewlink"}}<a href="/view/{{.}}">View {{.}}</a>{{end}}
{{define "filelink"}}<a href="/file/{{.Session}}/{{.FileIdentity.Hash}}">{{.FileIdentity.URI}}</a>{{end}}
`)).Funcs(template.FuncMap{
	"fuint64":  fuint64,
	"fuint32":  fuint32,
	"fcontent": fcontent,
	"localAddress": func(s string) string {
		// Try to translate loopback addresses to localhost, both for cosmetics and
		// because unspecified ipv6 addresses can break links on Windows.
		//
		// TODO(rfindley): In the future, it would be better not to assume the
		// server is running on localhost, and instead construct this address using
		// the remote host.
		host, port, err := net.SplitHostPort(s)
		if err != nil {
			return s
		}
		ip := net.ParseIP(host)
		if ip == nil {
			return s
		}
		if ip.IsLoopback() || ip.IsUnspecified() {
			return "localhost:" + port
		}
		return s
	},
	"options": func(s *cache.Session) []string {
		return showOptions(s.Options())
	},
})

var MainTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls server information{{end}}
{{define "body"}}
<h2>Caches</h2>
<ul>{{range .State.Caches}}<li>{{template "cachelink" .ID}}</li>{{end}}</ul>
<h2>Sessions</h2>
<ul>{{range .State.Sessions}}<li>{{template "sessionlink" .ID}} from {{template "cachelink" .Cache.ID}}</li>{{end}}</ul>
<h2>Views</h2>
<ul>{{range .State.Views}}<li>{{.Name}} is {{template "viewlink" .ID}} from {{template "sessionlink" .Session.ID}} in {{.Folder}}</li>{{end}}</ul>
<h2>Clients</h2>
<ul>{{range .State.Clients}}<li>{{template "clientlink" .Session.ID}}</li>{{end}}</ul>
<h2>Servers</h2>
<ul>{{range .State.Servers}}<li>{{template "serverlink" .ID}}</li>{{end}}</ul>
{{end}}
`))

var InfoTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls version information{{end}}
{{define "body"}}
{{.}}
{{end}}
`))

var MemoryTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls memory usage{{end}}
{{define "head"}}<meta http-equiv="refresh" content="5">{{end}}
{{define "body"}}
<h2>Stats</h2>
<table>
<tr><td class="label">Allocated bytes</td><td class="value">{{fuint64 .HeapAlloc}}</td></tr>
<tr><td class="label">Total allocated bytes</td><td class="value">{{fuint64 .TotalAlloc}}</td></tr>
<tr><td class="label">System bytes</td><td class="value">{{fuint64 .Sys}}</td></tr>
<tr><td class="label">Heap system bytes</td><td class="value">{{fuint64 .HeapSys}}</td></tr>
<tr><td class="label">Malloc calls</td><td class="value">{{fuint64 .Mallocs}}</td></tr>
<tr><td class="label">Frees</td><td class="value">{{fuint64 .Frees}}</td></tr>
<tr><td class="label">Idle heap bytes</td><td class="value">{{fuint64 .HeapIdle}}</td></tr>
<tr><td class="label">In use bytes</td><td class="value">{{fuint64 .HeapInuse}}</td></tr>
<tr><td class="label">Released to system bytes</td><td class="value">{{fuint64 .HeapReleased}}</td></tr>
<tr><td class="label">Heap object count</td><td class="value">{{fuint64 .HeapObjects}}</td></tr>
<tr><td class="label">Stack in use bytes</td><td class="value">{{fuint64 .StackInuse}}</td></tr>
<tr><td class="label">Stack from system bytes</td><td class="value">{{fuint64 .StackSys}}</td></tr>
<tr><td class="label">Bucket hash bytes</td><td class="value">{{fuint64 .BuckHashSys}}</td></tr>
<tr><td class="label">GC metadata bytes</td><td class="value">{{fuint64 .GCSys}}</td></tr>
<tr><td class="label">Off heap bytes</td><td class="value">{{fuint64 .OtherSys}}</td></tr>
</table>
<h2>By size</h2>
<table>
<tr><th>Size</th><th>Mallocs</th><th>Frees</th></tr>
{{range .BySize}}<tr><td class="value">{{fuint32 .Size}}</td><td class="value">{{fuint64 .Mallocs}}</td><td class="value">{{fuint64 .Frees}}</td></tr>{{end}}
</table>
{{end}}
`))

var DebugTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls Debug pages{{end}}
{{define "body"}}
<a href="/debug/pprof">Profiling</a>
{{end}}
`))

var CacheTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}Cache {{.ID}}{{end}}
{{define "body"}}
<h2>memoize.Store entries</h2>
<ul>{{range $k,$v := .MemStats}}<li>{{$k}} - {{$v}}</li>{{end}}</ul>
<h2>Per-package usage - not accurate, for guidance only</h2>
{{.PackageStats true}}
{{end}}
`))

var ClientTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}Client {{.Session.ID}}{{end}}
{{define "body"}}
Using session: <b>{{template "sessionlink" .Session.ID}}</b><br>
{{if .DebugAddress}}Debug this client at: <a href="http://{{localAddress .DebugAddress}}">{{localAddress .DebugAddress}}</a><br>{{end}}
Logfile: {{.Logfile}}<br>
Gopls Path: {{.GoplsPath}}<br>
<h2>Diagnostics</h2>
{{/*Service: []protocol.Server; each server has map[uri]fileReports;
	each fileReport: map[diagnosticSoure]diagnosticReport
	diagnosticSource is one of 5 source
	diagnosticReport: snapshotID and map[hash]*source.Diagnostic
	sourceDiagnostic: struct {
		Range    protocol.Range
		Message  string
		Source   string
		Code     string
		CodeHref string
		Severity protocol.DiagnosticSeverity
		Tags     []protocol.DiagnosticTag

		Related []RelatedInformation
	}
	RelatedInformation: struct {
		URI     span.URI
		Range   protocol.Range
		Message string
	}
	*/}}
<ul>{{range $k, $v := .Service.Diagnostics}}<li>{{$k}}:<ol>{{range $v}}<li>{{.}}</li>{{end}}</ol></li>{{end}}</ul>
{{end}}
`))

var ServerTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}Server {{.ID}}{{end}}
{{define "body"}}
{{if .DebugAddress}}Debug this server at: <a href="http://{{localAddress .DebugAddress}}">{{localAddress .DebugAddress}}</a><br>{{end}}
Logfile: {{.Logfile}}<br>
Gopls Path: {{.GoplsPath}}<br>
{{end}}
`))

var SessionTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}Session {{.ID}}{{end}}
{{define "body"}}
From: <b>{{template "cachelink" .Cache.ID}}</b><br>
<h2>Views</h2>
<ul>{{range .Views}}<li>{{.Name}} is {{template "viewlink" .ID}} in {{.Folder}}</li>{{end}}</ul>
<h2>Overlays</h2>
<ul>{{range .Overlays}}<li>{{template "filelink" .}}</li>{{end}}</ul>
<h2>Options</h2>
{{range options .}}<p>{{.}}{{end}}
{{end}}
`))

var ViewTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}View {{.ID}}{{end}}
{{define "body"}}
Name: <b>{{.Name}}</b><br>
Folder: <b>{{.Folder}}</b><br>
From: <b>{{template "sessionlink" .Session.ID}}</b><br>
<h2>Environment</h2>
<ul>{{range .Options.Env}}<li>{{.}}</li>{{end}}</ul>
{{end}}
`))

var FileTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}Overlay {{.FileIdentity.Hash}}{{end}}
{{define "body"}}
{{with .}}
	From: <b>{{template "sessionlink" .Session}}</b><br>
	URI: <b>{{.URI}}</b><br>
	Identifier: <b>{{.FileIdentity.Hash}}</b><br>
	Version: <b>{{.Version}}</b><br>
	Kind: <b>{{.Kind}}</b><br>
{{end}}
<h3>Contents</h3>
<pre>{{fcontent .Read}}</pre>
{{end}}
`))
