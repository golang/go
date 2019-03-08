# Generate Go types for the LSP protocol

## Setup

1. Make sure `node` is installed.
  * As explained at the [node site](https://nodejs.org Node)
  * You may need `node install @types/node` for the node runtime types
2. Install the typescript compiler, with `node install typescript`.
3. Make sure `tsc` and `node` are in your execution path.
4. Get the typescript code for the jsonrpc protocol with `git clone vscode-lanuageserver-node.git`

## Usage

```tsc go.ts && node go.js [-d dir] [-o out.go]```

and for simple checking

```gofmt -w out.go && golint out.go && go build out.go```

`-d dir` names the directory into which the `vscode-languageserver-node` repository was cloned.
It defaults to `$(HOME)`.

`-o out.go` says where the generated go code goes.
It defaults to `/tmp/tsprotocol.go`.

(The output file cannot yet be used to build `gopls`. That will be fixed in a future CL.)

## Note

`go.ts` uses the Typescript compiler's API, which is [introduced](https://github.com/Microsoft/TypeScript/wiki/Architectural-Overview API) in their wiki.