// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"context"
	"html/template"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof" // pull in the standard pprof handlers
)

func init() {
	http.HandleFunc("/", Render(mainTmpl, nil))
	http.HandleFunc("/debug/", Render(debugTmpl, nil))
}

// Serve starts and runs a debug server in the background.
// It also logs the port the server starts on, to allow for :0 auto assigned
// ports.
func Serve(ctx context.Context, addr string) error {
	if addr == "" {
		return nil
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	log.Printf("Debug serving on port: %d", listener.Addr().(*net.TCPAddr).Port)
	go func() {
		if err := http.Serve(listener, nil); err != nil {
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

var BaseTemplate = template.Must(template.New("").Parse(`
<html>
<head>
<title>{{template "title"}}</title>
<style>
.profile-name{
	display:inline-block;
	width:6rem;
}
</style>
</head>
<body>
{{template "title"}}
<br>
{{block "body" .Data}}
Unknown page
{{end}}
</body>
</html>
`))

var mainTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls server information{{end}}
{{define "body"}}
<A href="/debug/">Debug</A>
{{end}}
`))

var debugTmpl = template.Must(template.Must(BaseTemplate.Clone()).Parse(`
{{define "title"}}GoPls Debug pages{{end}}
{{define "body"}}
<A href="/debug/pprof">Profiling</A>
{{end}}
`))
