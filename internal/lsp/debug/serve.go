// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"bytes"
	"context"
	"go/token"
	"html/template"
	"log"
	"net"
	"net/http"
	"net/http/pprof"
	_ "net/http/pprof" // pull in the standard pprof handlers
	"path"
	"runtime"
	"strconv"
	"sync"

	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
)

type Cache interface {
	ID() string
	FileSet() *token.FileSet
}

type Session interface {
	ID() string
	Cache() Cache
	Files() []*File
	File(hash string) *File
}

type View interface {
	ID() string
	Name() string
	Folder() span.URI
	Session() Session
}

type File struct {
	Session Session
	URI     span.URI
	Data    string
	Error   error
	Hash    string
}

var (
	mu   sync.Mutex
	data = struct {
		Caches   []Cache
		Sessions []Session
		Views    []View
	}{}
)

// AddCache adds a cache to the set being served
func AddCache(cache Cache) {
	mu.Lock()
	defer mu.Unlock()
	data.Caches = append(data.Caches, cache)
}

// DropCache drops a cache from the set being served
func DropCache(cache Cache) {
	mu.Lock()
	defer mu.Unlock()
	//find and remove the cache
	if i, _ := findCache(cache.ID()); i >= 0 {
		copy(data.Caches[i:], data.Caches[i+1:])
		data.Caches[len(data.Caches)-1] = nil
		data.Caches = data.Caches[:len(data.Caches)-1]
	}
}

func findCache(id string) (int, Cache) {
	for i, c := range data.Caches {
		if c.ID() == id {
			return i, c
		}
	}
	return -1, nil
}

func getCache(r *http.Request) interface{} {
	mu.Lock()
	defer mu.Unlock()
	id := path.Base(r.URL.Path)
	result := struct {
		Cache
		Sessions []Session
	}{}
	_, result.Cache = findCache(id)

	// now find all the views that belong to this session
	for _, v := range data.Sessions {
		if v.Cache().ID() == id {
			result.Sessions = append(result.Sessions, v)
		}
	}
	return result
}

func findSession(id string) Session {
	for _, c := range data.Sessions {
		if c.ID() == id {
			return c
		}
	}
	return nil
}

func getSession(r *http.Request) interface{} {
	mu.Lock()
	defer mu.Unlock()
	id := path.Base(r.URL.Path)
	result := struct {
		Session
		Views []View
	}{
		Session: findSession(id),
	}
	// now find all the views that belong to this session
	for _, v := range data.Views {
		if v.Session().ID() == id {
			result.Views = append(result.Views, v)
		}
	}
	return result
}

func findView(id string) View {
	for _, c := range data.Views {
		if c.ID() == id {
			return c
		}
	}
	return nil
}

func getView(r *http.Request) interface{} {
	mu.Lock()
	defer mu.Unlock()
	id := path.Base(r.URL.Path)
	return findView(id)
}

func getFile(r *http.Request) interface{} {
	mu.Lock()
	defer mu.Unlock()
	hash := path.Base(r.URL.Path)
	sid := path.Base(path.Dir(r.URL.Path))
	session := findSession(sid)
	return session.File(hash)
}

func getInfo(r *http.Request) interface{} {
	buf := &bytes.Buffer{}
	PrintVersionInfo(buf, true, HTML)
	return template.HTML(buf.String())
}

func getMemory(r *http.Request) interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m
}

// AddSession adds a session to the set being served
func AddSession(session Session) {
	mu.Lock()
	defer mu.Unlock()
	data.Sessions = append(data.Sessions, session)
}

// DropSession drops a session from the set being served
func DropSession(session Session) {
	mu.Lock()
	defer mu.Unlock()
	//find and remove the session
}

// AddView adds a view to the set being served
func AddView(view View) {
	mu.Lock()
	defer mu.Unlock()
	data.Views = append(data.Views, view)
}

// DropView drops a view from the set being served
func DropView(view View) {
	mu.Lock()
	defer mu.Unlock()
	//find and remove the view
}

// Serve starts and runs a debug server in the background.
// It also logs the port the server starts on, to allow for :0 auto assigned
// ports.
func Serve(ctx context.Context, addr string) error {
	mu.Lock()
	defer mu.Unlock()
	if addr == "" {
		return nil
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	log.Printf("Debug serving on port: %d", listener.Addr().(*net.TCPAddr).Port)
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/", Render(mainTmpl, func(*http.Request) interface{} { return data }))
		mux.HandleFunc("/debug/", Render(debugTmpl, nil))
		telemetry.Handle(mux)
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		mux.HandleFunc("/cache/", Render(cacheTmpl, getCache))
		mux.HandleFunc("/session/", Render(sessionTmpl, getSession))
		mux.HandleFunc("/view/", Render(viewTmpl, getView))
		mux.HandleFunc("/file/", Render(fileTmpl, getFile))
		mux.HandleFunc("/info", Render(infoTmpl, getInfo))
		mux.HandleFunc("/memory", Render(memoryTmpl, getMemory))
		if err := http.Serve(listener, mux); err != nil {
			log.Printf("Debug server failed with %v", err)
			return
		}
		log.Printf("Debug server finished")
	}()
	return nil
}

func Render(tmpl *template.Template, fun func(*http.Request) interface{}) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		var data interface{}
		if fun != nil {
			data = fun(r)
		}
		if err := tmpl.Execute(w, data); err != nil {
			log.Print(err)
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
</style>
{{block "head" .}}{{end}}
</head>
<body>
<a href="/">Main</a>
<a href="/info">Info</a>
<a href="/memory">Memory</a>
<a href="/debug/">Debug</a>
<hr>
<h1>{{template "title" .}}</h1>
{{block "body" .}}
Unknown page
{{end}}
</body>
</html>

{{define "cachelink"}}<a href="/cache/{{.}}">Cache {{.}}</a>{{end}}
{{define "sessionlink"}}<a href="/session/{{.}}">Session {{.}}</a>{{end}}
{{define "viewlink"}}<a href="/view/{{.}}">View {{.}}</a>{{end}}
{{define "filelink"}}<a href="/file/{{.Session.ID}}/{{.Hash}}">{{.URI}}</a>{{end}}
`)).Funcs(template.FuncMap{
	"fuint64": fuint64,
	"fuint32": fuint32,
})

var mainTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls server information{{end}}
{{define "body"}}
<h2>Caches</h2>
<ul>{{range .Caches}}<li>{{template "cachelink" .ID}}</li>{{end}}</ul>
<h2>Sessions</h2>
<ul>{{range .Sessions}}<li>{{template "sessionlink" .ID}} from {{template "cachelink" .Cache.ID}}</li>{{end}}</ul>
<h2>Views</h2>
<ul>{{range .Views}}<li>{{.Name}} is {{template "viewlink" .ID}} from {{template "sessionlink" .Session.ID}} in {{.Folder}}</li>{{end}}</ul>
{{end}}
`))

var infoTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls version information{{end}}
{{define "body"}}
{{.}}
{{end}}
`))

var memoryTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
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
<tr><td class="label">GC metaata bytes</td><td class="value">{{fuint64 .GCSys}}</td></tr>
<tr><td class="label">Off heap bytes</td><td class="value">{{fuint64 .OtherSys}}</td></tr>
</table>
<h2>By size</h2>
<table>
<tr><th>Size</th><th>Mallocs</th><th>Frees</th></tr>
{{range .BySize}}<tr><td class="value">{{fuint32 .Size}}</td><td class="value">{{fuint64 .Mallocs}}</td><td class="value">{{fuint64 .Frees}}</td></tr>{{end}}
</table>
{{end}}
`))

var debugTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls Debug pages{{end}}
{{define "body"}}
<a href="/debug/pprof">Profiling</a>
<a href="/debug/rpcz">RPCz</a>
<a href="/debug/tracez">Tracez</a>
{{end}}
`))

var cacheTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}Cache {{.ID}}{{end}}
{{define "body"}}
<h2>Sessions</h2>
<ul>{{range .Sessions}}<li>{{template "sessionlink" .ID}}</li>{{end}}</ul>
{{end}}
`))

var sessionTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}Session {{.ID}}{{end}}
{{define "body"}}
From: <b>{{template "cachelink" .Cache.ID}}</b><br>
<h2>Views</h2>
<ul>{{range .Views}}<li>{{.Name}} is {{template "viewlink" .ID}} in {{.Folder}}</li>{{end}}</ul>
<h2>Files</h2>
<ul>{{range .Files}}<li>{{template "filelink" .}}</li>{{end}}</ul>
{{end}}
`))

var viewTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}View {{.ID}}{{end}}
{{define "body"}}
Name: <b>{{.Name}}</b><br>
Folder: <b>{{.Folder}}</b><br>
From: <b>{{template "sessionlink" .Session.ID}}</b><br>
<h2>Environment</h2>
<ul>{{range .Env}}<li>{{.}}</li>{{end}}</ul>
{{end}}
`))

var fileTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
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
