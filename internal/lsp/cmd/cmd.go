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
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/lsprpc"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
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

	// the options configuring function to invoke when building a server
	options func(*source.Options)

	// The name of the binary, used in help and telemetry.
	name string

	// The working directory to run commands in.
	wd string

	// The environment variables to use.
	env []string

	// Support for remote LSP server.
	Remote string `flag:"remote" help:"forward all commands to a remote lsp specified by this flag. With no special prefix, this is assumed to be a TCP address. If prefixed by 'unix;', the subsequent address is assumed to be a unix domain socket. If 'auto', or prefixed by 'auto;', the remote address is automatically resolved based on the executing environment."`

	// Verbose enables verbose logging.
	Verbose bool `flag:"v" help:"verbose output"`

	// VeryVerbose enables a higher level of verbosity in logging output.
	VeryVerbose bool `flag:"vv" help:"very verbose output"`

	// Control ocagent export of telemetry
	OCAgent string `flag:"ocagent" help:"the address of the ocagent (e.g. http://localhost:55678), or off"`

	// PrepareOptions is called to update the options when a new view is built.
	// It is primarily to allow the behavior of gopls to be modified by hooks.
	PrepareOptions func(*source.Options)
}

func (app *Application) verbose() bool {
	return app.Verbose || app.VeryVerbose
}

// New returns a new Application ready to run.
func New(name, wd string, env []string, options func(*source.Options)) *Application {
	if wd == "" {
		wd, _ = os.Getwd()
	}
	app := &Application{
		options: options,
		name:    name,
		wd:      wd,
		env:     env,
		OCAgent: "off", //TODO: Remove this line to default the exporter to on

		Serve: Serve{
			RemoteListenTimeout: 1 * time.Minute,
		},
	}
	return app
}

// Name implements tool.Application returning the binary name.
func (app *Application) Name() string { return app.name }

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
gopls is a Go language server. It is typically used with an editor to provide
language features. When no command is specified, gopls will default to the 'serve'
command. The language features can also be accessed via the gopls command-line interface.

Available commands are:
`)
	fmt.Fprint(f.Output(), `
main:
`)
	for _, c := range app.mainCommands() {
		fmt.Fprintf(f.Output(), "  %s : %v\n", c.Name(), c.ShortHelp())
	}
	fmt.Fprint(f.Output(), `
features:
`)
	for _, c := range app.featureCommands() {
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
	ctx = debug.WithInstance(ctx, app.wd, app.OCAgent)
	app.Serve.app = app
	if len(args) == 0 {
		return tool.Run(ctx, &app.Serve, args)
	}
	command, args := args[0], args[1:]
	for _, c := range app.commands() {
		if c.Name() == command {
			return tool.Run(ctx, c, args)
		}
	}
	return tool.CommandLineErrorf("Unknown command %v", command)
}

// commands returns the set of commands supported by the gopls tool on the
// command line.
// The command is specified by the first non flag argument.
func (app *Application) commands() []tool.Application {
	var commands []tool.Application
	commands = append(commands, app.mainCommands()...)
	commands = append(commands, app.featureCommands()...)
	return commands
}

func (app *Application) mainCommands() []tool.Application {
	return []tool.Application{
		&app.Serve,
		&version{app: app},
		&bug{},
		&apiJSON{},
		&licenses{app: app},
	}
}

func (app *Application) featureCommands() []tool.Application {
	return []tool.Application{
		&callHierarchy{app: app},
		&check{app: app},
		&definition{app: app},
		&foldingRanges{app: app},
		&format{app: app},
		&highlight{app: app},
		&implementation{app: app},
		&imports{app: app},
		newRemote(app, ""),
		newRemote(app, "inspect"),
		&links{app: app},
		&prepareRename{app: app},
		&references{app: app},
		&rename{app: app},
		&semtok{app: app},
		&signature{app: app},
		&suggestedFix{app: app},
		&symbols{app: app},
		newWorkspace(app),
		&workspaceSymbol{app: app},
	}
}

var (
	internalMu          sync.Mutex
	internalConnections = make(map[string]*connection)
)

func (app *Application) connect(ctx context.Context) (*connection, error) {
	switch {
	case app.Remote == "":
		connection := newConnection(app)
		connection.Server = lsp.NewServer(cache.New(app.options).NewSession(ctx), connection.Client)
		ctx = protocol.WithClient(ctx, connection.Client)
		return connection, connection.initialize(ctx, app.options)
	case strings.HasPrefix(app.Remote, "internal@"):
		internalMu.Lock()
		defer internalMu.Unlock()
		opts := source.DefaultOptions().Clone()
		if app.options != nil {
			app.options(opts)
		}
		key := fmt.Sprintf("%s %v %v %v", app.wd, opts.PreferredContentFormat, opts.HierarchicalDocumentSymbolSupport, opts.SymbolMatcher)
		if c := internalConnections[key]; c != nil {
			return c, nil
		}
		remote := app.Remote[len("internal@"):]
		ctx := xcontext.Detach(ctx) //TODO:a way of shutting down the internal server
		connection, err := app.connectRemote(ctx, remote)
		if err != nil {
			return nil, err
		}
		internalConnections[key] = connection
		return connection, nil
	default:
		return app.connectRemote(ctx, app.Remote)
	}
}

// CloseTestConnections terminates shared connections used in command tests. It
// should only be called from tests.
func CloseTestConnections(ctx context.Context) {
	for _, c := range internalConnections {
		c.Shutdown(ctx)
		c.Exit(ctx)
	}
}

func (app *Application) connectRemote(ctx context.Context, remote string) (*connection, error) {
	connection := newConnection(app)
	conn, err := lsprpc.ConnectToRemote(ctx, remote)
	if err != nil {
		return nil, err
	}
	stream := jsonrpc2.NewHeaderStream(conn)
	cc := jsonrpc2.NewConn(stream)
	connection.Server = protocol.ServerDispatcher(cc)
	ctx = protocol.WithClient(ctx, connection.Client)
	cc.Go(ctx,
		protocol.Handlers(
			protocol.ClientHandler(connection.Client,
				jsonrpc2.MethodNotFound)))
	return connection, connection.initialize(ctx, app.options)
}

var matcherString = map[source.SymbolMatcher]string{
	source.SymbolFuzzy:           "fuzzy",
	source.SymbolCaseSensitive:   "caseSensitive",
	source.SymbolCaseInsensitive: "caseInsensitive",
}

func (c *connection) initialize(ctx context.Context, options func(*source.Options)) error {
	params := &protocol.ParamInitialize{}
	params.RootURI = protocol.URIFromPath(c.Client.app.wd)
	params.Capabilities.Workspace.Configuration = true

	// Make sure to respect configured options when sending initialize request.
	opts := source.DefaultOptions().Clone()
	if options != nil {
		options(opts)
	}
	// If you add an additional option here, you must update the map key in connect.
	params.Capabilities.TextDocument.Hover = protocol.HoverClientCapabilities{
		ContentFormat: []protocol.MarkupKind{opts.PreferredContentFormat},
	}
	params.Capabilities.TextDocument.DocumentSymbol.HierarchicalDocumentSymbolSupport = opts.HierarchicalDocumentSymbolSupport
	params.Capabilities.TextDocument.SemanticTokens = protocol.SemanticTokensClientCapabilities{}
	params.Capabilities.TextDocument.SemanticTokens.Formats = []string{"relative"}
	params.Capabilities.TextDocument.SemanticTokens.Requests.Range = true
	params.Capabilities.TextDocument.SemanticTokens.Requests.Full = true
	params.Capabilities.TextDocument.SemanticTokens.TokenTypes = lsp.SemanticTypes()
	params.Capabilities.TextDocument.SemanticTokens.TokenModifiers = lsp.SemanticModifiers()
	params.InitializationOptions = map[string]interface{}{
		"symbolMatcher": matcherString[opts.SymbolMatcher],
	}
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

	diagnosticsMu   sync.Mutex
	diagnosticsDone chan struct{}

	filesMu sync.Mutex
	files   map[span.URI]*cmdFile
}

type cmdFile struct {
	uri         span.URI
	mapper      *protocol.ColumnMapper
	err         error
	added       bool
	diagnostics []protocol.Diagnostic
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

// fileURI converts a DocumentURI to a file:// span.URI, panicking if it's not a file.
func fileURI(uri protocol.DocumentURI) span.URI {
	sURI := uri.SpanURI()
	if !sURI.IsFile() {
		panic(fmt.Sprintf("%q is not a file URI", uri))
	}
	return sURI
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
		if c.app.verbose() {
			log.Print("Info:", p.Message)
		}
	case protocol.Log:
		if c.app.verbose() {
			log.Print("Log:", p.Message)
		}
	default:
		if c.app.verbose() {
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

func (c *cmdClient) Configuration(ctx context.Context, p *protocol.ParamConfiguration) ([]interface{}, error) {
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
		m := map[string]interface{}{
			"env": env,
			"analyses": map[string]bool{
				"fillreturns":    true,
				"nonewvars":      true,
				"noresultvalues": true,
				"undeclaredname": true,
			},
		}
		if c.app.VeryVerbose {
			m["verboseOutput"] = true
		}
		results[i] = m
	}
	return results, nil
}

func (c *cmdClient) ApplyEdit(ctx context.Context, p *protocol.ApplyWorkspaceEditParams) (*protocol.ApplyWorkspaceEditResult, error) {
	return &protocol.ApplyWorkspaceEditResult{Applied: false, FailureReason: "not implemented"}, nil
}

func (c *cmdClient) PublishDiagnostics(ctx context.Context, p *protocol.PublishDiagnosticsParams) error {
	if p.URI == "gopls://diagnostics-done" {
		close(c.diagnosticsDone)
	}
	// Don't worry about diagnostics without versions.
	if p.Version == 0 {
		return nil
	}

	c.filesMu.Lock()
	defer c.filesMu.Unlock()

	file := c.getFile(ctx, fileURI(p.URI))
	file.diagnostics = p.Diagnostics
	return nil
}

func (c *cmdClient) Progress(context.Context, *protocol.ProgressParams) error {
	return nil
}

func (c *cmdClient) ShowDocument(context.Context, *protocol.ShowDocumentParams) (*protocol.ShowDocumentResult, error) {
	return nil, nil
}

func (c *cmdClient) WorkDoneProgressCreate(context.Context, *protocol.WorkDoneProgressCreateParams) error {
	return nil
}

func (c *cmdClient) getFile(ctx context.Context, uri span.URI) *cmdFile {
	file, found := c.files[uri]
	if !found || file.err != nil {
		file = &cmdFile{
			uri: uri,
		}
		c.files[uri] = file
	}
	if file.mapper == nil {
		fname := uri.Filename()
		content, err := ioutil.ReadFile(fname)
		if err != nil {
			file.err = errors.Errorf("getFile: %v: %v", uri, err)
			return file
		}
		f := c.fset.AddFile(fname, -1, len(content))
		f.SetLinesForContent(content)
		converter := span.NewContentConverter(fname, content)
		file.mapper = &protocol.ColumnMapper{
			URI:       uri,
			Converter: converter,
			Content:   content,
		}
	}
	return file
}

func (c *connection) AddFile(ctx context.Context, uri span.URI) *cmdFile {
	c.Client.filesMu.Lock()
	defer c.Client.filesMu.Unlock()

	file := c.Client.getFile(ctx, uri)
	// This should never happen.
	if file == nil {
		return &cmdFile{
			uri: uri,
			err: fmt.Errorf("no file found for %s", uri),
		}
	}
	if file.err != nil || file.added {
		return file
	}
	file.added = true
	p := &protocol.DidOpenTextDocumentParams{
		TextDocument: protocol.TextDocumentItem{
			URI:        protocol.URIFromSpanURI(uri),
			LanguageID: "go",
			Version:    1,
			Text:       string(file.mapper.Content),
		},
	}
	if err := c.Server.DidOpen(ctx, p); err != nil {
		file.err = errors.Errorf("%v: %v", uri, err)
	}
	return file
}

func (c *connection) semanticTokens(ctx context.Context, p *protocol.SemanticTokensRangeParams) (*protocol.SemanticTokens, error) {
	// use range to avoid limits on full
	resp, err := c.Server.SemanticTokensRange(ctx, p)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (c *connection) diagnoseFiles(ctx context.Context, files []span.URI) error {
	var untypedFiles []interface{}
	for _, file := range files {
		untypedFiles = append(untypedFiles, string(file))
	}
	c.Client.diagnosticsMu.Lock()
	defer c.Client.diagnosticsMu.Unlock()

	c.Client.diagnosticsDone = make(chan struct{})
	_, err := c.Server.NonstandardRequest(ctx, "gopls/diagnoseFiles", map[string]interface{}{"files": untypedFiles})
	if err != nil {
		close(c.Client.diagnosticsDone)
		return err
	}

	<-c.Client.diagnosticsDone
	return nil
}

func (c *connection) terminate(ctx context.Context) {
	if strings.HasPrefix(c.Client.app.Remote, "internal@") {
		// internal connections need to be left alive for the next test
		return
	}
	//TODO: do we need to handle errors on these calls?
	c.Shutdown(ctx)
	//TODO: right now calling exit terminates the process, we should rethink that
	//server.Exit(ctx)
}

// Implement io.Closer.
func (c *cmdClient) Close() error {
	return nil
}
