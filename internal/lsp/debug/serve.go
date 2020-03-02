// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"bytes"
	"context"
	"fmt"
	"go/token"
	"html/template"
	"io"
	stdlog "log"
	"net"
	"net/http"
	"net/http/pprof"
	_ "net/http/pprof" // pull in the standard pprof handlers
	"os"
	"path"
	"path/filepath"
	"reflect"
	"runtime"
	rpprof "runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export"
	"golang.org/x/tools/internal/telemetry/export/ocagent"
	"golang.org/x/tools/internal/telemetry/export/prometheus"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
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

	ocagent    *ocagent.Exporter
	prometheus *prometheus.Exporter
	rpcs       *rpcs
	traces     *traces
	State      *State
}

// State holds debugging information related to the server state.
type State struct {
	mu       sync.Mutex
	caches   objset
	sessions objset
	views    objset
	clients  objset
	servers  objset
}

type ider interface {
	ID() string
}

type objset struct {
	objs []ider
}

func (s *objset) add(elem ider) {
	s.objs = append(s.objs, elem)
}

func (s *objset) drop(elem ider) {
	var newobjs []ider
	for _, obj := range s.objs {
		if obj.ID() != elem.ID() {
			newobjs = append(newobjs, obj)
		}
	}
	s.objs = newobjs
}

func (s *objset) find(id string) ider {
	for _, e := range s.objs {
		if e.ID() == id {
			return e
		}
	}
	return nil
}

// Caches returns the set of Cache objects currently being served.
func (st *State) Caches() []Cache {
	st.mu.Lock()
	defer st.mu.Unlock()
	caches := make([]Cache, len(st.caches.objs))
	for i, c := range st.caches.objs {
		caches[i] = c.(Cache)
	}
	return caches
}

// Sessions returns the set of Session objects currently being served.
func (st *State) Sessions() []Session {
	st.mu.Lock()
	defer st.mu.Unlock()
	sessions := make([]Session, len(st.sessions.objs))
	for i, s := range st.sessions.objs {
		sessions[i] = s.(Session)
	}
	return sessions
}

// Views returns the set of View objects currently being served.
func (st *State) Views() []View {
	st.mu.Lock()
	defer st.mu.Unlock()
	views := make([]View, len(st.views.objs))
	for i, v := range st.views.objs {
		views[i] = v.(View)
	}
	return views
}

// Clients returns the set of Clients currently being served.
func (st *State) Clients() []Client {
	st.mu.Lock()
	defer st.mu.Unlock()
	clients := make([]Client, len(st.clients.objs))
	for i, c := range st.clients.objs {
		clients[i] = c.(Client)
	}
	return clients
}

// Servers returns the set of Servers the instance is currently connected to.
func (st *State) Servers() []Server {
	st.mu.Lock()
	defer st.mu.Unlock()
	servers := make([]Server, len(st.servers.objs))
	for i, s := range st.servers.objs {
		servers[i] = s.(Server)
	}
	return servers
}

// A Client is an incoming connection from a remote client.
type Client interface {
	ID() string
	Session() Session
	DebugAddress() string
	Logfile() string
	ServerID() string
}

// A Server is an outgoing connection to a remote LSP server.
type Server interface {
	ID() string
	DebugAddress() string
	Logfile() string
	ClientID() string
}

// A Cache is an in-memory cache.
type Cache interface {
	ID() string
	FileSet() *token.FileSet
	MemStats() map[reflect.Type]int
}

// A Session is an LSP serving session.
type Session interface {
	ID() string
	Cache() Cache
	Files() []*File
	File(hash string) *File
}

// A View is a root directory within a Session.
type View interface {
	ID() string
	Name() string
	Folder() span.URI
	Session() Session
}

// A File is is a file within a session.
type File struct {
	Session Session
	URI     span.URI
	Data    string
	Error   error
	Hash    string
}

// AddCache adds a cache to the set being served.
func (st *State) AddCache(cache Cache) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.caches.add(cache)
}

// DropCache drops a cache from the set being served.
func (st *State) DropCache(cache Cache) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.caches.drop(cache)
}

// AddSession adds a session to the set being served.
func (st *State) AddSession(session Session) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.sessions.add(session)
}

// DropSession drops a session from the set being served.
func (st *State) DropSession(session Session) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.sessions.drop(session)
}

// AddView adds a view to the set being served.
func (st *State) AddView(view View) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.views.add(view)
}

// DropView drops a view from the set being served.
func (st *State) DropView(view View) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.views.drop(view)
}

// AddClient adds a client to the set being served.
func (st *State) AddClient(client Client) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.clients.add(client)
}

// DropClient adds a client to the set being served.
func (st *State) DropClient(client Client) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.clients.drop(client)
}

// AddServer adds a server to the set being queried. In practice, there should
// be at most one remote server.
func (st *State) AddServer(server Server) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.servers.add(server)
}

// DropServer drops a server to the set being queried.
func (st *State) DropServer(server Server) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.servers.drop(server)
}

func (i *Instance) getCache(r *http.Request) interface{} {
	i.State.mu.Lock()
	defer i.State.mu.Unlock()
	id := path.Base(r.URL.Path)
	c, ok := i.State.caches.find(id).(Cache)
	if !ok {
		return nil
	}
	result := struct {
		Cache
		Sessions []Session
	}{
		Cache: c,
	}

	// now find all the views that belong to this session
	for _, vd := range i.State.sessions.objs {
		v := vd.(Session)
		if v.Cache().ID() == id {
			result.Sessions = append(result.Sessions, v)
		}
	}
	return result
}

func (i *Instance) getSession(r *http.Request) interface{} {
	i.State.mu.Lock()
	defer i.State.mu.Unlock()
	id := path.Base(r.URL.Path)
	s, ok := i.State.sessions.find(id).(Session)
	if !ok {
		return nil
	}
	result := struct {
		Session
		Views []View
	}{
		Session: s,
	}
	// now find all the views that belong to this session
	for _, vd := range i.State.views.objs {
		v := vd.(View)
		if v.Session().ID() == id {
			result.Views = append(result.Views, v)
		}
	}
	return result
}

func (i Instance) getClient(r *http.Request) interface{} {
	i.State.mu.Lock()
	defer i.State.mu.Unlock()
	id := path.Base(r.URL.Path)
	c, ok := i.State.clients.find(id).(Client)
	if !ok {
		return nil
	}
	return c
}

func (i Instance) getServer(r *http.Request) interface{} {
	i.State.mu.Lock()
	defer i.State.mu.Unlock()
	id := path.Base(r.URL.Path)
	s, ok := i.State.servers.find(id).(Server)
	if !ok {
		return nil
	}
	return s
}

func (i Instance) getView(r *http.Request) interface{} {
	i.State.mu.Lock()
	defer i.State.mu.Unlock()
	id := path.Base(r.URL.Path)
	v, ok := i.State.views.find(id).(View)
	if !ok {
		return nil
	}
	return v
}

func (i *Instance) getFile(r *http.Request) interface{} {
	i.State.mu.Lock()
	defer i.State.mu.Unlock()
	hash := path.Base(r.URL.Path)
	sid := path.Base(path.Dir(r.URL.Path))
	s, ok := i.State.sessions.find(sid).(Session)
	if !ok {
		return nil
	}
	return s.File(hash)
}

func (i *Instance) getInfo(r *http.Request) interface{} {
	buf := &bytes.Buffer{}
	i.PrintServerInfo(buf)
	return template.HTML(buf.String())
}

func getMemory(r *http.Request) interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m
}

// NewInstance creates debug instance ready for use using the supplied configuration.
func NewInstance(workdir, agent string) *Instance {
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
	i.rpcs = &rpcs{}
	i.traces = &traces{}
	i.State = &State{}
	export.SetExporter(i)
	return i
}

// SetLogFile sets the logfile for use with this instance.
func (i *Instance) SetLogFile(logfile string) (func(), error) {
	// TODO: probably a better solution for deferring closure to the caller would
	// be for the debug instance to itself be closed, but this fixes the
	// immediate bug of logs not being captured.
	closeLog := func() {}
	if logfile != "" {
		if logfile == "auto" {
			logfile = filepath.Join(os.TempDir(), fmt.Sprintf("gopls-%d.log", os.Getpid()))
		}
		f, err := os.Create(logfile)
		if err != nil {
			return nil, fmt.Errorf("unable to create log file: %v", err)
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
		stdlog.Printf("debug server listening on port %d", port)
	}
	log.Print(ctx, "Debug serving", tag.Of("Port", port))
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/", render(mainTmpl, func(*http.Request) interface{} { return i }))
		mux.HandleFunc("/debug/", render(debugTmpl, nil))
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		if i.prometheus != nil {
			mux.HandleFunc("/metrics/", i.prometheus.Serve)
		}
		if i.rpcs != nil {
			mux.HandleFunc("/rpc/", render(rpcTmpl, i.rpcs.getData))
		}
		if i.traces != nil {
			mux.HandleFunc("/trace/", render(traceTmpl, i.traces.getData))
		}
		mux.HandleFunc("/cache/", render(cacheTmpl, i.getCache))
		mux.HandleFunc("/session/", render(sessionTmpl, i.getSession))
		mux.HandleFunc("/view/", render(viewTmpl, i.getView))
		mux.HandleFunc("/client/", render(clientTmpl, i.getClient))
		mux.HandleFunc("/server/", render(serverTmpl, i.getServer))
		mux.HandleFunc("/file/", render(fileTmpl, i.getFile))
		mux.HandleFunc("/info", render(infoTmpl, i.getInfo))
		mux.HandleFunc("/memory", render(memoryTmpl, getMemory))
		if err := http.Serve(listener, mux); err != nil {
			log.Error(ctx, "Debug server failed", err)
			return
		}
		log.Print(ctx, "Debug server finished")
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
			i.writeMemoryDebug(nextThresholdGiB)
			log.Print(ctx, fmt.Sprintf("Wrote memory usage debug info to %v", os.TempDir()))
			nextThresholdGiB++
		}
	}()
}

func (i *Instance) writeMemoryDebug(threshold uint64) error {
	fname := func(t string) string {
		return fmt.Sprintf("gopls.%d-%dGiB-%s", os.Getpid(), threshold, t)
	}

	f, err := os.Create(filepath.Join(os.TempDir(), fname("heap.pb.gz")))
	if err != nil {
		return err
	}
	defer f.Close()
	if err := rpprof.Lookup("heap").WriteTo(f, 0); err != nil {
		return err
	}

	f, err = os.Create(filepath.Join(os.TempDir(), fname("goroutines.txt")))
	if err != nil {
		return err
	}
	defer f.Close()
	if err := rpprof.Lookup("goroutine").WriteTo(f, 1); err != nil {
		return err
	}
	return nil
}

func (i *Instance) StartSpan(ctx context.Context, spn *telemetry.Span) {
	if i.ocagent != nil {
		i.ocagent.StartSpan(ctx, spn)
	}
	if i.traces != nil {
		i.traces.StartSpan(ctx, spn)
	}
}

func (i *Instance) FinishSpan(ctx context.Context, spn *telemetry.Span) {
	if i.ocagent != nil {
		i.ocagent.FinishSpan(ctx, spn)
	}
	if i.traces != nil {
		i.traces.FinishSpan(ctx, spn)
	}
}

//TODO: remove this hack
// capture stderr at startup because it gets modified in a way that this
// logger should not respect
var stderr = os.Stderr

func (i *Instance) Log(ctx context.Context, event telemetry.Event) {
	if event.Error != nil {
		fmt.Fprintf(stderr, "%v\n", event)
	}
	protocol.LogEvent(ctx, event)
	if i.ocagent != nil {
		i.ocagent.Log(ctx, event)
	}
}

func (i *Instance) Metric(ctx context.Context, data telemetry.MetricData) {
	if i.ocagent != nil {
		i.ocagent.Metric(ctx, data)
	}
	if i.traces != nil {
		i.prometheus.Metric(ctx, data)
	}
	if i.rpcs != nil {
		i.rpcs.Metric(ctx, data)
	}
}

type dataFunc func(*http.Request) interface{}

func render(tmpl *template.Template, fun dataFunc) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		var data interface{}
		if fun != nil {
			data = fun(r)
		}
		if err := tmpl.Execute(w, data); err != nil {
			log.Error(context.Background(), "", err)
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

var baseTemplate = template.Must(template.New("").Parse(`
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
{{define "filelink"}}<a href="/file/{{.Session.ID}}/{{.Hash}}">{{.URI}}</a>{{end}}
`)).Funcs(template.FuncMap{
	"fuint64": fuint64,
	"fuint32": fuint32,
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
})

var mainTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}GoPls server information{{end}}
{{define "body"}}
<h2>Caches</h2>
<ul>{{range .State.Caches}}<li>{{template "cachelink" .ID}}</li>{{end}}</ul>
<h2>Sessions</h2>
<ul>{{range .State.Sessions}}<li>{{template "sessionlink" .ID}} from {{template "cachelink" .Cache.ID}}</li>{{end}}</ul>
<h2>Views</h2>
<ul>{{range .State.Views}}<li>{{.Name}} is {{template "viewlink" .ID}} from {{template "sessionlink" .Session.ID}} in {{.Folder}}</li>{{end}}</ul>
<h2>Clients</h2>
<ul>{{range .State.Clients}}<li>{{template "clientlink" .ID}}</li>{{end}}</ul>
<h2>Servers</h2>
<ul>{{range .State.Servers}}<li>{{template "serverlink" .ID}}</li>{{end}}</ul>
{{end}}
`))

var infoTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}GoPls version information{{end}}
{{define "body"}}
{{.}}
{{end}}
`))

var memoryTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
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

var debugTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}GoPls Debug pages{{end}}
{{define "body"}}
<a href="/debug/pprof">Profiling</a>
{{end}}
`))

var cacheTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}Cache {{.ID}}{{end}}
{{define "body"}}
<h2>Sessions</h2>
<ul>{{range .Sessions}}<li>{{template "sessionlink" .ID}}</li>{{end}}</ul>
<h2>memoize.Store entries</h2>
<ul>{{range $k,$v := .MemStats}}<li>{{$k}} - {{$v}}</li>{{end}}</ul>
{{end}}
`))

var clientTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}Client {{.ID}}{{end}}
{{define "body"}}
Using session: <b>{{template "sessionlink" .Session.ID}}</b><br>
{{if .DebugAddress}}Debug this client at: <a href="http://{{localAddress .DebugAddress}}">{{localAddress .DebugAddress}}</a><br>{{end}}
Logfile: {{.Logfile}}<br>
Gopls Path: {{.GoplsPath}}<br>
{{end}}
`))

var serverTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}Server {{.ID}}{{end}}
{{define "body"}}
{{if .DebugAddress}}Debug this server at: <a href="http://{{localAddress .DebugAddress}}">{{localAddress .DebugAddress}}</a><br>{{end}}
Logfile: {{.Logfile}}<br>
Gopls Path: {{.GoplsPath}}<br>
{{end}}
`))

var sessionTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}Session {{.ID}}{{end}}
{{define "body"}}
From: <b>{{template "cachelink" .Cache.ID}}</b><br>
<h2>Views</h2>
<ul>{{range .Views}}<li>{{.Name}} is {{template "viewlink" .ID}} in {{.Folder}}</li>{{end}}</ul>
<h2>Files</h2>
<ul>{{range .Files}}<li>{{template "filelink" .}}</li>{{end}}</ul>
{{end}}
`))

var viewTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}View {{.ID}}{{end}}
{{define "body"}}
Name: <b>{{.Name}}</b><br>
Folder: <b>{{.Folder}}</b><br>
From: <b>{{template "sessionlink" .Session.ID}}</b><br>
<h2>Environment</h2>
<ul>{{range .Env}}<li>{{.}}</li>{{end}}</ul>
{{end}}
`))

var fileTmpl = template.Must(template.Must(baseTemplate.Clone()).Parse(`
{{define "title"}}File {{.Hash}}{{end}}
{{define "body"}}
From: <b>{{template "sessionlink" .Session.ID}}</b><br>
URI: <b>{{.URI}}</b><br>
Hash: <b>{{.Hash}}</b><br>
Error: <b>{{.Error}}</b><br>
<h3>Contents</h3>
<pre>{{.Data}}</pre>
{{end}}
`))
