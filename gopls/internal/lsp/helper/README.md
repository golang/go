# Generate server_gen.go

`helper` generates the file `../server_gen.go` (in package
`internal/lsp`) which contains stub declarations of server methods.

To invoke it, run `go generate` in the `gopls/internal/lsp` directory.

It is derived from `gopls/internal/lsp/protocol/tsserver.go`, which
itself is generated from the protocol downloaded from VSCode, so be
sure to run `go generate` in the protocol first. Or run `go generate
./...` twice in the gopls directory.

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
