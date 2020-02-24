# Status

gopls is currently in **alpha**, so it is **not stable**.

gopls is currently under active development by the Go team. The code is in the [x/tools] repository, in [golang.org/x/tools/internal/lsp] and [golang.org/x/tools/gopls].

## Supported features

<!--- TODO: supported features
details and status for the features
missing features
--->

### Autocompletion
### Jump to definition
### Signature help
### Hover
### Document symbols
### References
### Rename

## Known issues

1. Editing multiple modules in one editor window: [#32394]
1. Type checking does not work in cgo packages: [#35721]
1. Does not work with build tags: [#29202]
1. Find references and rename only work in a single package: [#32877]

[x/tools]: https://github.com/golang/tools
[golang.org/x/tools/gopls]: https://github.com/golang/tools/tree/master/gopls
[golang.org/x/tools/internal/lsp]: https://github.com/golang/tools/tree/master/internal/lsp


[#32394]: https://github.com/golang/go/issues/32394
[#35721]: https://github.com/golang/go/issues/35721
[#29202]: https://github.com/golang/go/issues/29202
[#32877]: https://github.com/golang/go/issues/32877
