# gopls implementation documentation

This is not intended as a complete description of the implementation, for the most the part the package godoc, code comments and the code itself hold that.
Instead this is meant to be a guide into finding parts of the implementation, and understanding some core concepts used throughout the implementation.

## View/Session/Cache

Throughout the code there are references to these three concepts, and they build on each other.

At the base is the *Cache*. This is the level at which we hold information that is global in nature, for instance information about the file system and its contents.

Above that is the *Session*, which holds information for a connection to an editor. This layer hold things like the edited files (referred to as overlays).

The top layer is called the *View*. This holds the configuration, and the mapping to configured packages.

The purpose of this layering is to allow a single editor session to have multiple views active whilst still sharing as much information as possible for efficiency.
In theory if only the View layer existed, the results would be identical, but slower and using more memory.

## Code location

gopls will be developed in the [x/tools] Go repository; the core packages are in [internal/lsp], and the binary and integration tests are located in [gopls].

Below is a list of the core packages of gopls, and their primary purpose:

Package | Description
--- | ---
[gopls] | the main binary, plugins and integration tests
[internal/lsp] | the core message handling package
[internal/lsp/cache] | the cache layer
[internal/lsp/cmd] | the gopls command line layer
[internal/lsp/debug] | features to aid in debugging gopls
[internal/lsp/protocol] | the lsp protocol layer and wire format
[internal/lsp/source] | the core feature implementations
[internal/span] | a package for dealing with source file locations
[internal/memoize] | a function invocation cache used to reduce the work done
[internal/jsonrpc2] | an implementation of the JSON RPC2 specification

[gopls]: https://github.com/golang/tools/tree/master/gopls
[internal/jsonrpc2]: https://github.com/golang/tools/tree/master/internal/jsonrpc2
[internal/lsp]: https://github.com/golang/tools/tree/master/internal/lsp
[internal/lsp/cache]: https://github.com/golang/tools/tree/master/internal/lsp/cache
[internal/lsp/cmd]: https://github.com/golang/tools/tree/master/internal/lsp/cmd
[internal/lsp/debug]: https://github.com/golang/tools/tree/master/internal/lsp/debug
[internal/lsp/protocol]: https://github.com/golang/tools/tree/master/internal/lsp/protocol
[internal/lsp/source]: https://github.com/golang/tools/tree/master/internal/lsp/source
[internal/memoize]: https://github.com/golang/tools/tree/master/internal/memoize
[internal/span]: https://github.com/golang/tools/tree/master/internal/span
[x/tools]: https://github.com/golang/tools

