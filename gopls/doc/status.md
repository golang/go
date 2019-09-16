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

1. Cursor resets to the beginning or end of file on format: [#31937]
1. Editing multiple modules in one editor window: [#32394]
1. Language features do not work with cgo: [#32898]
1. Does not work with build tags: [#29202]
1. Find references and rename only work in a single package: [#32869], [#32877]
1. Completion does not work well after go or defer statements: [#29313]
1. Changes in files outside of the editor are not yet tracked: [#31553]

[x/tools]: https://github.com/golang/tools
[golang.org/x/tools/gopls]: https://github.com/golang/tools/tree/master/gopls
[golang.org/x/tools/internal/lsp]: https://github.com/golang/tools/tree/master/internal/lsp


[#31937]: https://github.com/golang/go/issues/31937
[#32394]: https://github.com/golang/go/issues/32394
[#32898]: https://github.com/golang/go/issues/32898
[#29202]: https://github.com/golang/go/issues/29202
[#32869]: https://github.com/golang/go/issues/32869
[#32877]: https://github.com/golang/go/issues/32877
[#29313]: https://github.com/golang/go/issues/29313
[#31553]: https://github.com/golang/go/issues/31553


