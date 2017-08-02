# godoc

This directory contains most of the code for running a godoc server. The
executable lives at golang.org/x/tools/cmd/godoc.

## Development mode

In production, CSS/JS/template assets need to be compiled into the godoc
binary. It can be tedious to recompile assets every time, but you can pass a
flag to load CSS/JS/templates from disk every time a page loads:

```
godoc -templates=$GOPATH/src/golang.org/x/tools/godoc/static -http=:6060
```

## Recompiling static assets

The files that live at `static/style.css`, `static/jquery.js` and so on are not
present in the final binary. They are placed into `static/static.go` by running
`go generate`. So to compile a change and test it in your browser:

1) Make changes to e.g. `static/style.css`.

2) Run `go generate golang.org/x/tools/godoc/static` so `static/static.go` picks
up the change.

3) Run `go install golang.org/x/tools/cmd/godoc` so the compiled `godoc` binary
picks up the change.

4) Run `godoc -http=:6060` and view your changes in the browser. You may need
to disable your browser's cache to avoid reloading a stale file.
