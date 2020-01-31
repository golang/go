# Generate server_gen.go

`helper` generates boilerplate code for server.go by processing the
generated code in `protocol/tsserver.go`.

First, build `helper` in this directory (`go build .`).

In directory `lsp`, executing `go generate server.go` generates the stylized file
`server_gen.go` that contains stubs for type `Server`.

It decides what stubs are needed and their signatures
by looking at the `Server` interface (`-t` flag). These all look somewhat like
`Resolve(context.Context, *CompletionItem) (*CompletionItem, error)`.

It then parses the `lsp` directory (`-u` flag) to see if there is a corresponding
implementation function (which in this case would be named `resolve`). If so
it discovers the parameter names needed, and generates (in `server_gen.go`) code
like

``` go
func (s *Server) resolve(ctx context.Context, params *protocol.CompletionItem) (*protocol.CompletionItem, error) {
    return s.resolve(ctx, params)
}
```

If `resolve` is not defined (and it is not), then the body of the generated function is

```go
    return nil, notImplemented("resolve")
```

So to add a capability currently not implemented, just define it somewhere in `lsp`.
In this case, just define `func (s *Server) resolve(...)` and re-generate `server_gen.go`.
