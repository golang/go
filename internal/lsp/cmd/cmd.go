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
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"net"
	"os"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/protocol"
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

	// An initial, common go/packages configuration
	Config packages.Config

	// Support for remote lsp server
	Remote string `flag:"remote" help:"*EXPERIMENTAL* - forward all commands to a remote lsp"`
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
	if app.Config.Dir == "" {
		if wd, err := os.Getwd(); err == nil {
			app.Config.Dir = wd
		}
	}
	app.Config.Mode = packages.LoadSyntax
	app.Config.Tests = true
	if app.Config.Fset == nil {
		app.Config.Fset = token.NewFileSet()
	}
	app.Config.Context = ctx
	app.Config.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
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
		&query{app: app},
		&check{app: app},
	}
}

type cmdClient interface {
	protocol.Client

	prepare(app *Application, server protocol.Server)
}

func (app *Application) connect(ctx context.Context, client cmdClient) (protocol.Server, error) {
	var server protocol.Server
	switch app.Remote {
	case "":
		server = lsp.NewClientServer(client)
	case "internal":
		cr, sw, _ := os.Pipe()
		sr, cw, _ := os.Pipe()
		var jc *jsonrpc2.Conn
		jc, server, _ = protocol.NewClient(jsonrpc2.NewHeaderStream(cr, cw), client)
		go jc.Run(ctx)
		go lsp.NewServer(jsonrpc2.NewHeaderStream(sr, sw)).Run(ctx)
	default:
		conn, err := net.Dial("tcp", app.Remote)
		if err != nil {
			return nil, err
		}
		stream := jsonrpc2.NewHeaderStream(conn, conn)
		var jc *jsonrpc2.Conn
		jc, server, _ = protocol.NewClient(stream, client)
		if err != nil {
			return nil, err
		}
		go jc.Run(ctx)
	}
	params := &protocol.InitializeParams{}
	params.RootURI = string(span.FileURI(app.Config.Dir))
	params.Capabilities.Workspace.Configuration = true
	client.prepare(app, server)
	if _, err := server.Initialize(ctx, params); err != nil {
		return nil, err
	}
	if err := server.Initialized(ctx, &protocol.InitializedParams{}); err != nil {
		return nil, err
	}
	return server, nil
}

type baseClient struct {
	protocol.Server
	app    *Application
	server protocol.Server
	fset   *token.FileSet
}

func (c *baseClient) ShowMessage(ctx context.Context, p *protocol.ShowMessageParams) error { return nil }
func (c *baseClient) ShowMessageRequest(ctx context.Context, p *protocol.ShowMessageRequestParams) (*protocol.MessageActionItem, error) {
	return nil, nil
}
func (c *baseClient) LogMessage(ctx context.Context, p *protocol.LogMessageParams) error {
	switch p.Type {
	case protocol.Error:
		log.Print("Error:", p.Message)
	case protocol.Warning:
		log.Print("Warning:", p.Message)
	case protocol.Info:
		log.Print("Info:", p.Message)
	case protocol.Log:
		log.Print("Log:", p.Message)
	default:
		log.Print(p.Message)
	}
	return nil
}
func (c *baseClient) Telemetry(ctx context.Context, t interface{}) error { return nil }
func (c *baseClient) RegisterCapability(ctx context.Context, p *protocol.RegistrationParams) error {
	return nil
}
func (c *baseClient) UnregisterCapability(ctx context.Context, p *protocol.UnregistrationParams) error {
	return nil
}
func (c *baseClient) WorkspaceFolders(ctx context.Context) ([]protocol.WorkspaceFolder, error) {
	return nil, nil
}
func (c *baseClient) Configuration(ctx context.Context, p *protocol.ConfigurationParams) ([]interface{}, error) {
	results := make([]interface{}, len(p.Items))
	for i, item := range p.Items {
		if item.Section != "gopls" {
			continue
		}
		env := map[string]interface{}{}
		for _, value := range c.app.Config.Env {
			l := strings.SplitN(value, "=", 2)
			if len(l) != 2 {
				continue
			}
			env[l[0]] = l[1]
		}
		results[i] = map[string]interface{}{"env": env}
	}
	return results, nil
}
func (c *baseClient) ApplyEdit(ctx context.Context, p *protocol.ApplyWorkspaceEditParams) (bool, error) {
	return false, nil
}
func (c *baseClient) PublishDiagnostics(ctx context.Context, p *protocol.PublishDiagnosticsParams) error {
	return nil
}

func (c *baseClient) prepare(app *Application, server protocol.Server) {
	c.app = app
	c.server = server
	c.fset = token.NewFileSet()
}

func (c *baseClient) AddFile(ctx context.Context, uri span.URI) (*protocol.ColumnMapper, error) {
	fname, err := uri.Filename()
	if err != nil {
		return nil, fmt.Errorf("%v: %v", uri, err)
	}
	content, err := ioutil.ReadFile(fname)
	if err != nil {
		return nil, fmt.Errorf("%v: %v", uri, err)
	}
	f := c.fset.AddFile(fname, -1, len(content))
	f.SetLinesForContent(content)
	m := protocol.NewColumnMapper(uri, c.fset, f, content)
	p := &protocol.DidOpenTextDocumentParams{}
	p.TextDocument.URI = string(uri)
	p.TextDocument.Text = string(content)
	if err := c.server.DidOpen(ctx, p); err != nil {
		return nil, fmt.Errorf("%v: %v", uri, err)
	}
	return m, nil
}
