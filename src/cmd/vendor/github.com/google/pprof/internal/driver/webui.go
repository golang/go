// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package driver

import (
	"bytes"
	"fmt"
	"html/template"
	"io"
	"net"
	"net/http"
	gourl "net/url"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/google/pprof/internal/graph"
	"github.com/google/pprof/internal/measurement"
	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/internal/report"
	"github.com/google/pprof/profile"
)

// webInterface holds the state needed for serving a browser based interface.
type webInterface struct {
	prof         *profile.Profile
	copier       profileCopier
	options      *plugin.Options
	help         map[string]string
	settingsFile string
}

func makeWebInterface(p *profile.Profile, copier profileCopier, opt *plugin.Options) (*webInterface, error) {
	settingsFile, err := settingsFileName()
	if err != nil {
		return nil, err
	}
	return &webInterface{
		prof:         p,
		copier:       copier,
		options:      opt,
		help:         make(map[string]string),
		settingsFile: settingsFile,
	}, nil
}

// maxEntries is the maximum number of entries to print for text interfaces.
const maxEntries = 50

// errorCatcher is a UI that captures errors for reporting to the browser.
type errorCatcher struct {
	plugin.UI
	errors []string
}

func (ec *errorCatcher) PrintErr(args ...interface{}) {
	ec.errors = append(ec.errors, strings.TrimSuffix(fmt.Sprintln(args...), "\n"))
	ec.UI.PrintErr(args...)
}

// webArgs contains arguments passed to templates in webhtml.go.
type webArgs struct {
	Title       string
	Errors      []string
	Total       int64
	SampleTypes []string
	Legend      []string
	DocURL      string
	Standalone  bool // True for command-line generation of HTML
	Help        map[string]string
	Nodes       []string
	HTMLBody    template.HTML
	TextBody    string
	Top         []report.TextItem
	Listing     report.WebListData
	FlameGraph  template.JS
	Stacks      template.JS
	Configs     []configMenuEntry
	UnitDefs    []measurement.UnitType
}

func serveWebInterface(hostport string, p *profile.Profile, o *plugin.Options, disableBrowser bool) error {
	host, port, err := getHostAndPort(hostport)
	if err != nil {
		return err
	}
	interactiveMode = true
	copier := makeProfileCopier(p)
	ui, err := makeWebInterface(p, copier, o)
	if err != nil {
		return err
	}
	for n, c := range pprofCommands {
		ui.help[n] = c.description
	}
	for n, help := range configHelp {
		ui.help[n] = help
	}
	ui.help["details"] = "Show information about the profile and this view"
	ui.help["graph"] = "Display profile as a directed graph"
	ui.help["flamegraph"] = "Display profile as a flame graph"
	ui.help["reset"] = "Show the entire profile"
	ui.help["save_config"] = "Save current settings"

	server := o.HTTPServer
	if server == nil {
		server = defaultWebServer
	}
	args := &plugin.HTTPServerArgs{
		Hostport: net.JoinHostPort(host, strconv.Itoa(port)),
		Host:     host,
		Port:     port,
		Handlers: map[string]http.Handler{
			"/":              http.HandlerFunc(ui.dot),
			"/top":           http.HandlerFunc(ui.top),
			"/disasm":        http.HandlerFunc(ui.disasm),
			"/source":        http.HandlerFunc(ui.source),
			"/peek":          http.HandlerFunc(ui.peek),
			"/flamegraph":    http.HandlerFunc(ui.stackView),
			"/flamegraph2":   redirectWithQuery("flamegraph", http.StatusMovedPermanently), // Keep legacy URL working.
			"/flamegraphold": redirectWithQuery("flamegraph", http.StatusMovedPermanently), // Keep legacy URL working.
			"/saveconfig":    http.HandlerFunc(ui.saveConfig),
			"/deleteconfig":  http.HandlerFunc(ui.deleteConfig),
			"/download": http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				w.Header().Set("Content-Type", "application/vnd.google.protobuf+gzip")
				w.Header().Set("Content-Disposition", "attachment;filename=profile.pb.gz")
				p.Write(w)
			}),
		},
	}

	url := "http://" + args.Hostport

	o.UI.Print("Serving web UI on ", url)

	if o.UI.WantBrowser() && !disableBrowser {
		go openBrowser(url, o)
	}
	return server(args)
}

func getHostAndPort(hostport string) (string, int, error) {
	host, portStr, err := net.SplitHostPort(hostport)
	if err != nil {
		return "", 0, fmt.Errorf("could not split http address: %v", err)
	}
	if host == "" {
		host = "localhost"
	}
	var port int
	if portStr == "" {
		ln, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
		if err != nil {
			return "", 0, fmt.Errorf("could not generate random port: %v", err)
		}
		port = ln.Addr().(*net.TCPAddr).Port
		err = ln.Close()
		if err != nil {
			return "", 0, fmt.Errorf("could not generate random port: %v", err)
		}
	} else {
		port, err = strconv.Atoi(portStr)
		if err != nil {
			return "", 0, fmt.Errorf("invalid port number: %v", err)
		}
	}
	return host, port, nil
}
func defaultWebServer(args *plugin.HTTPServerArgs) error {
	ln, err := net.Listen("tcp", args.Hostport)
	if err != nil {
		return err
	}
	isLocal := isLocalhost(args.Host)
	handler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if isLocal {
			// Only allow local clients
			host, _, err := net.SplitHostPort(req.RemoteAddr)
			if err != nil || !isLocalhost(host) {
				http.Error(w, "permission denied", http.StatusForbidden)
				return
			}
		}
		h := args.Handlers[req.URL.Path]
		if h == nil {
			// Fall back to default behavior
			h = http.DefaultServeMux
		}
		h.ServeHTTP(w, req)
	})

	// We serve the ui at /ui/ and redirect there from the root. This is done
	// to surface any problems with serving the ui at a non-root early. See:
	//
	// https://github.com/google/pprof/pull/348
	mux := http.NewServeMux()
	mux.Handle("/ui/", http.StripPrefix("/ui", handler))
	mux.Handle("/", redirectWithQuery("/ui", http.StatusTemporaryRedirect))
	s := &http.Server{Handler: mux}
	return s.Serve(ln)
}

// redirectWithQuery responds with a given redirect code, preserving query
// parameters in the redirect URL. It does not convert relative paths to
// absolute paths like http.Redirect does, so that HTTPServerArgs.Handlers can
// generate relative redirects that work with the external prefixing.
func redirectWithQuery(path string, code int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		pathWithQuery := &gourl.URL{Path: path, RawQuery: r.URL.RawQuery}
		w.Header().Set("Location", pathWithQuery.String())
		w.WriteHeader(code)
	}
}

func isLocalhost(host string) bool {
	for _, v := range []string{"localhost", "127.0.0.1", "[::1]", "::1"} {
		if host == v {
			return true
		}
	}
	return false
}

func openBrowser(url string, o *plugin.Options) {
	// Construct URL.
	baseURL, _ := gourl.Parse(url)
	current := currentConfig()
	u, _ := current.makeURL(*baseURL)

	// Give server a little time to get ready.
	time.Sleep(time.Millisecond * 500)

	for _, b := range browsers() {
		args := strings.Split(b, " ")
		if len(args) == 0 {
			continue
		}
		viewer := exec.Command(args[0], append(args[1:], u.String())...)
		viewer.Stderr = os.Stderr
		if err := viewer.Start(); err == nil {
			return
		}
	}
	// No visualizer succeeded, so just print URL.
	o.UI.PrintErr(u.String())
}

// makeReport generates a report for the specified command.
// If configEditor is not null, it is used to edit the config used for the report.
func (ui *webInterface) makeReport(w http.ResponseWriter, req *http.Request,
	cmd []string, configEditor func(*config)) (*report.Report, []string) {
	cfg := currentConfig()
	if err := cfg.applyURL(req.URL.Query()); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		ui.options.UI.PrintErr(err)
		return nil, nil
	}
	if configEditor != nil {
		configEditor(&cfg)
	}
	catcher := &errorCatcher{UI: ui.options.UI}
	options := *ui.options
	options.UI = catcher
	_, rpt, err := generateRawReport(ui.copier.newCopy(), cmd, cfg, &options)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		ui.options.UI.PrintErr(err)
		return nil, nil
	}
	return rpt, catcher.errors
}

// renderHTML generates html using the named template based on the contents of data.
func renderHTML(dst io.Writer, tmpl string, rpt *report.Report, errList, legend []string, data webArgs) error {
	file := getFromLegend(legend, "File: ", "unknown")
	profile := getFromLegend(legend, "Type: ", "unknown")
	data.Title = file + " " + profile
	data.Errors = errList
	data.Total = rpt.Total()
	data.DocURL = rpt.DocURL()
	data.Legend = legend
	return getHTMLTemplates().ExecuteTemplate(dst, tmpl, data)
}

// render responds with html generated by passing data to the named template.
func (ui *webInterface) render(w http.ResponseWriter, req *http.Request, tmpl string,
	rpt *report.Report, errList, legend []string, data webArgs) {
	data.SampleTypes = sampleTypes(ui.prof)
	data.Help = ui.help
	data.Configs = configMenu(ui.settingsFile, *req.URL)
	html := &bytes.Buffer{}
	if err := renderHTML(html, tmpl, rpt, errList, legend, data); err != nil {
		http.Error(w, "internal template error", http.StatusInternalServerError)
		ui.options.UI.PrintErr(err)
		return
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(html.Bytes())
}

// dot generates a web page containing an svg diagram.
func (ui *webInterface) dot(w http.ResponseWriter, req *http.Request) {
	rpt, errList := ui.makeReport(w, req, []string{"svg"}, nil)
	if rpt == nil {
		return // error already reported
	}

	// Generate dot graph.
	g, config := report.GetDOT(rpt)
	legend := config.Labels
	config.Labels = nil
	dot := &bytes.Buffer{}
	graph.ComposeDot(dot, g, &graph.DotAttributes{}, config)

	// Convert to svg.
	svg, err := dotToSvg(dot.Bytes())
	if err != nil {
		http.Error(w, "Could not execute dot; may need to install graphviz.",
			http.StatusNotImplemented)
		ui.options.UI.PrintErr("Failed to execute dot. Is Graphviz installed?\n", err)
		return
	}

	// Get all node names into an array.
	nodes := []string{""} // dot starts with node numbered 1
	for _, n := range g.Nodes {
		nodes = append(nodes, n.Info.Name)
	}

	ui.render(w, req, "graph", rpt, errList, legend, webArgs{
		HTMLBody: template.HTML(string(svg)),
		Nodes:    nodes,
	})
}

func dotToSvg(dot []byte) ([]byte, error) {
	cmd := exec.Command("dot", "-Tsvg")
	out := &bytes.Buffer{}
	cmd.Stdin, cmd.Stdout, cmd.Stderr = bytes.NewBuffer(dot), out, os.Stderr
	if err := cmd.Run(); err != nil {
		return nil, err
	}

	// Fix dot bug related to unquoted ampersands.
	svg := bytes.Replace(out.Bytes(), []byte("&;"), []byte("&amp;;"), -1)

	// Cleanup for embedding by dropping stuff before the <svg> start.
	if pos := bytes.Index(svg, []byte("<svg")); pos >= 0 {
		svg = svg[pos:]
	}
	return svg, nil
}

func (ui *webInterface) top(w http.ResponseWriter, req *http.Request) {
	rpt, errList := ui.makeReport(w, req, []string{"top"}, func(cfg *config) {
		cfg.NodeCount = 500
	})
	if rpt == nil {
		return // error already reported
	}
	top, legend := report.TextItems(rpt)
	var nodes []string
	for _, item := range top {
		nodes = append(nodes, item.Name)
	}

	ui.render(w, req, "top", rpt, errList, legend, webArgs{
		Top:   top,
		Nodes: nodes,
	})
}

// disasm generates a web page containing disassembly.
func (ui *webInterface) disasm(w http.ResponseWriter, req *http.Request) {
	args := []string{"disasm", req.URL.Query().Get("f")}
	rpt, errList := ui.makeReport(w, req, args, nil)
	if rpt == nil {
		return // error already reported
	}

	out := &bytes.Buffer{}
	if err := report.PrintAssembly(out, rpt, ui.options.Obj, maxEntries); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		ui.options.UI.PrintErr(err)
		return
	}

	legend := report.ProfileLabels(rpt)
	ui.render(w, req, "plaintext", rpt, errList, legend, webArgs{
		TextBody: out.String(),
	})

}

// source generates a web page containing source code annotated with profile
// data.
func (ui *webInterface) source(w http.ResponseWriter, req *http.Request) {
	args := []string{"weblist", req.URL.Query().Get("f")}
	rpt, errList := ui.makeReport(w, req, args, nil)
	if rpt == nil {
		return // error already reported
	}

	// Generate source listing.
	listing, err := report.MakeWebList(rpt, ui.options.Obj, maxEntries)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		ui.options.UI.PrintErr(err)
		return
	}

	legend := report.ProfileLabels(rpt)
	ui.render(w, req, "sourcelisting", rpt, errList, legend, webArgs{
		Listing: listing,
	})
}

// peek generates a web page listing callers/callers.
func (ui *webInterface) peek(w http.ResponseWriter, req *http.Request) {
	args := []string{"peek", req.URL.Query().Get("f")}
	rpt, errList := ui.makeReport(w, req, args, func(cfg *config) {
		cfg.Granularity = "lines"
	})
	if rpt == nil {
		return // error already reported
	}

	out := &bytes.Buffer{}
	if err := report.Generate(out, rpt, ui.options.Obj); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		ui.options.UI.PrintErr(err)
		return
	}

	legend := report.ProfileLabels(rpt)
	ui.render(w, req, "plaintext", rpt, errList, legend, webArgs{
		TextBody: out.String(),
	})
}

// saveConfig saves URL configuration.
func (ui *webInterface) saveConfig(w http.ResponseWriter, req *http.Request) {
	if err := setConfig(ui.settingsFile, *req.URL); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		ui.options.UI.PrintErr(err)
		return
	}
}

// deleteConfig deletes a configuration.
func (ui *webInterface) deleteConfig(w http.ResponseWriter, req *http.Request) {
	name := req.URL.Query().Get("config")
	if err := removeConfig(ui.settingsFile, name); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		ui.options.UI.PrintErr(err)
		return
	}
}

// getFromLegend returns the suffix of an entry in legend that starts
// with param.  It returns def if no such entry is found.
func getFromLegend(legend []string, param, def string) string {
	for _, s := range legend {
		if strings.HasPrefix(s, param) {
			return s[len(param):]
		}
	}
	return def
}
