// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cmd handles the gopls command line.
// It contains a handler for each of the modes, along with all the flag handling
// and the command line output format.
package cmd

import (
	"context"
	"flag"
	"fmt"
	"go/token"
	"io/ioutil"
	"log"
	"net"
	"os"
	"strings"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

// Application is the main application as passed to tool.Main
// It handles the main command line parsing and dispatch to the sub commands.
type Application struct {
	// Core application flags

	// Embed the basic profiling flags supported by the tool package
	tool.Profile

	// We include the server configuration directly for now, so the flags work
	// even without the verb.
	// TODO: Remove this when we stop allowing the serve verb by default.
	Serve Serve

	// The base cache to use for sessions from this application.
	cache source.Cache

	// The working directory to run commands in.
	wd string

	// The environment variables to use.
	env []string

	// Support for remote lsp server
	Remote string `flag:"remote" help:"*EXPERIMENTAL* - forward all commands to a remote lsp"`

	// Enable verbose logging
	Verbose bool `flag:"v" help:"Verbose output"`
}

// Returns a new Application ready to run.
func New(wd string, env []string) *Application {
	if wd == "" {
		wd, _ = os.Getwd()
	}
	app := &Application{
		cache: cache.New(),
		wd:    wd,
		env:   env,
	}
	return app
}

// Name implements tool.Application returning the binary name.
func (app *Application) Name() string { return "gopls" }

// Usage implements tool.Application returning empty extra argument usage.
func (app *Application) Usage() string { return "<command> [command-flags] [command-args]" }

// ShortHelp implements tool.Application returning the main binary help.
func (app *Application) ShortHelp() string {
	return "The Go Language source tools."
}

// DetailedHelp implements tool.Application returning the main binary help.
// This includes the short help for all the sub commands.
func (app *Application) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Available commands are:
`)
	for _, c := range app.commands() {
		fmt.Fprintf(f.Output(), "  %s : %v\n", c.Name(), c.ShortHelp())
	}
	fmt.Fprint(f.Output(), `
gopls flags are:
`)
	f.PrintDefaults()
}

// Run takes the args after top level flag processing, and invokes the correct
// sub command as specified by the first argument.
// If no arguments are passed it will invoke the server sub command, as a
// temporary measure for compatibility.
func (app *Application) Run(ctx context.Context, args ...string) error {
	app.Serve.app = app
	if len(args) == 0 {
		tool.Main(ctx, &app.Serve, args)
		return nil
	}
	command, args := args[0], args[1:]
	for _, c := range app.commands() {
		if c.Name() == command {
			tool.Main(ctx, c, args)
			return nil
		}
	}
	return tool.CommandLineErrorf("Unknown command %v", command)
}

// commands returns the set of commands supported by the gopls tool on the
// command line.
// The command is specified by the first non flag argument.
func (app *Application) commands() []tool.Application {
	return []tool.Application{
		&app.Serve,
		&bug{},
		&check{app: app},
		&format{app: app},
		&query{app: app},
		&version{app: app},
	}
}

var (
	internalMu          sync.Mutex
	internalConnections = make(map[string]*connection)
)

func (app *Application) connect(ctx context.Context) (*connection, error) {
	switch app.Remote {
	case "":
		connection := newConnection(app)
		connection.Server = lsp.NewClientServer(app.cache, connection.Client)
		return connection, connection.initialize(ctx)
	case "internal":
		internalMu.Lock()
		defer internalMu.Unlock()
		if c := internalConnections[app.wd]; c != nil {
			return c, nil
		}
		connection := newConnection(app)
		ctx := context.Background() //TODO:a way of shutting down the internal server
		cr, sw, _ := os.Pipe()
		sr, cw, _ := os.Pipe()
		var jc *jsonrpc2.Conn
		jc, connection.Server, _ = protocol.NewClient(jsonrpc2.NewHeaderStream(cr, cw), connection.Client)
		go jc.Run(ctx)
		go lsp.NewServer(app.cache, jsonrpc2.NewHeaderStream(sr, sw)).Run(ctx)
		if err := connection.initialize(ctx); err != nil {
			return nil, err
		}
		internalConnections[app.wd] = connection
		return connection, nil
	default:
		connection := newConnection(app)
		conn, err := net.Dial("tcp", app.Remote)
		if err != nil {
			return nil, err
		}
		stream := jsonrpc2.NewHeaderStream(conn, conn)
		var jc *jsonrpc2.Conn
		jc, connection.Server, _ = protocol.NewClient(stream, connection.Client)
		go jc.Run(ctx)
		return connection, connection.initialize(ctx)
	}
}

func (c *connection) initialize(ctx context.Context) error {
	params := &protocol.InitializeParams{}
	params.RootURI = string(span.FileURI(c.Client.app.wd))
	params.Capabilities.Workspace.Configuration = true
	params.Capabilities.TextDocument.Hover.ContentFormat = []protocol.MarkupKind{protocol.PlainText}
	if _, err := c.Server.Initialize(ctx, params); err != nil {
		return err
	}
	if err := c.Server.Initialized(ctx, &protocol.InitializedParams{}); err != nil {
		return err
	}
	return nil
}

type connection struct {
	protocol.Server
	Client *cmdClient
}

type cmdClient struct {
	protocol.Server
	app  *Application
	fset *token.FileSet

	filesMu sync.Mutex
	files   map[span.URI]*cmdFile
}

type cmdFile struct {
	uri            span.URI
	mapper         *protocol.ColumnMapper
	err            error
	added          bool
	hasDiagnostics chan struct{}
	diagnosticsMu  sync.Mutex
	diagnostics    []protocol.Diagnostic
}

func newConnection(app *Application) *connection {
	return &connection{
		Client: &cmdClient{
			app:   app,
			fset:  token.NewFileSet(),
			files: make(map[span.URI]*cmdFile),
		},
	}
}

func (c *cmdClient) ShowMessage(ctx context.Context, p *protocol.ShowMessageParams) error { return nil }

func (c *cmdClient) ShowMessageRequest(ctx context.Context, p *protocol.ShowMessageRequestParams) (*protocol.MessageActionItem, error) {
	return nil, nil
}

func (c *cmdClient) LogMessage(ctx context.Context, p *protocol.LogMessageParams) error {
	switch p.Type {
	case protocol.Error:
		log.Print("Error:", p.Message)
	case protocol.Warning:
		log.Print("Warning:", p.Message)
	case protocol.Info:
		if c.app.Verbose {
			log.Print("Info:", p.Message)
		}
	case protocol.Log:
		if c.app.Verbose {
			log.Print("Log:", p.Message)
		}
	default:
		if c.app.Verbose {
			log.Print(p.Message)
		}
	}
	return nil
}

func (c *cmdClient) Event(ctx context.Context, t *interface{}) error { return nil }

func (c *cmdClient) RegisterCapability(ctx context.Context, p *protocol.RegistrationParams) error {
	return nil
}

func (c *cmdClient) UnregisterCapability(ctx context.Context, p *protocol.UnregistrationParams) error {
	return nil
}

func (c *cmdClient) WorkspaceFolders(ctx context.Context) ([]protocol.WorkspaceFolder, error) {
	return nil, nil
}

func (c *cmdClient) Configuration(ctx context.Context, p *protocol.ConfigurationParams) ([]interface{}, error) {
	results := make([]interface{}, len(p.Items))
	for i, item := range p.Items {
		if item.Section != "gopls" {
			continue
		}
		env := map[string]interface{}{}
		for _, value := range c.app.env {
			l := strings.SplitN(value, "=", 2)
			if len(l) != 2 {
				continue
			}
			env[l[0]] = l[1]
		}
		results[i] = map[string]interface{}{
			"env":           env,
			"noDocsOnHover": true,
		}
	}
	return results, nil
}

func (c *cmdClient) ApplyEdit(ctx context.Context, p *protocol.ApplyWorkspaceEditParams) (*protocol.ApplyWorkspaceEditResponse, error) {
	return &protocol.ApplyWorkspaceEditResponse{Applied: false, FailureReason: "not implemented"}, nil
}

func (c *cmdClient) PublishDiagnostics(ctx context.Context, p *protocol.PublishDiagnosticsParams) error {
	c.filesMu.Lock()
	defer c.filesMu.Unlock()
	uri := span.URI(p.URI)
	file := c.getFile(ctx, uri)
	file.diagnosticsMu.Lock()
	defer file.diagnosticsMu.Unlock()
	hadDiagnostics := file.diagnostics != nil
	file.diagnostics = p.Diagnostics
	if file.diagnostics == nil {
		file.diagnostics = []protocol.Diagnostic{}
	}
	if !hadDiagnostics {
		close(file.hasDiagnostics)
	}
	return nil
}

func (c *cmdClient) getFile(ctx context.Context, uri span.URI) *cmdFile {
	file, found := c.files[uri]
	if !found {
		file = &cmdFile{
			uri:            uri,
			hasDiagnostics: make(chan struct{}),
		}
		c.files[uri] = file
	}
	if file.mapper == nil {
		fname := uri.Filename()
		content, err := ioutil.ReadFile(fname)
		if err != nil {
			file.err = fmt.Errorf("%v: %v", uri, err)
			return file
		}
		f := c.fset.AddFile(fname, -1, len(content))
		f.SetLinesForContent(content)
		file.mapper = protocol.NewColumnMapper(uri, fname, c.fset, f, content)
	}
	return file
}

func (c *connection) AddFile(ctx context.Context, uri span.URI) *cmdFile {
	c.Client.filesMu.Lock()
	defer c.Client.filesMu.Unlock()
	file := c.Client.getFile(ctx, uri)
	if !file.added {
		file.added = true
		p := &protocol.DidOpenTextDocumentParams{}
		p.TextDocument.URI = string(uri)
		p.TextDocument.Text = string(file.mapper.Content)
		if err := c.Server.DidOpen(ctx, p); err != nil {
			file.err = fmt.Errorf("%v: %v", uri, err)
		}
	}
	return file
}

func (c *connection) terminate(ctx context.Context) {
	if c.Client.app.Remote == "internal" {
		// internal connections need to be left alive for the next test
		return
	}
	//TODO: do we need to handle errors on these calls?
	c.Shutdown(ctx)
	//TODO: right now calling exit terminates the process, we should rethink that
	//server.Exit(ctx)
}
