# Go Telemetry

This repository holds the Go Telemetry server code and libraries, used for
hosting [telemetry.go.dev](https://telemetry.go.dev) and instrumenting Go
toolchain programs with opt-in telemetry.

**Warning**: this repository is intended for use only in tools maintained by
the Go team, including tools in the Go distribution and auxiliary tools like
[gopls](https://pkg.go.dev/golang.org/x/tools/gopls) or
[govulncheck](https://pkg.go.dev/golang.org/x/vuln/cmd/govulncheck). There are
no compatibility guarantees for any of the packages here: public APIs will
change in breaking ways as the telemetry integration is refined.

## Notable Packages

- The [x/telemetry/counter](https://pkg.go.dev/golang.org/x/telemetry/counter)
  package provides a library for instrumenting programs with counters and stack
  reports.
- The [x/telemetry/upload](https://pkg.go.dev/golang.org/x/telemetry/upload)
  package provides a hook for Go toolchain programs to upload telemetry data,
  if the user has opted in to telemetry uploading.
- The [x/telemetry/cmd/gotelemetry](https://pkg.go.dev/pkg/golang.org/x/telemetry/cmd/gotelemetry)
  command is used for managing telemetry data and configuration.
- The [x/telemetry/config](https://pkg.go.dev/pkg/golang.org/x/telemetry/config)
  package defines the subset of telemetry data that has been approved for
  uploading by the telemetry proposal process.
- The [x/telemetry/godev](https://pkg.go.dev/pkg/golang.org/x/telemetry/godev) directory defines
  the services running at [telemetry.go.dev](https://telemetry.go.dev).

## Contributing

This repository uses Gerrit for code changes. To learn how to submit changes to
this repository, see https://golang.org/doc/contribute.html.

The main issue tracker for the time repository is located at
https://github.com/golang/go/issues. Prefix your issue with "x/telemetry:" in
the subject line, so it is easy to find.

### Linting & Formatting

This repository uses [eslint](https://eslint.org/) to format TS files,
[stylelint](https://stylelint.io/) to format CSS files, and
[prettier](https://prettier.io/) to format TS, CSS, Markdown, and YAML files.

See the style guides:

- [TypeScript](https://google.github.io/styleguide/tsguide.html)
- [CSS](https://go.dev/wiki/CSSStyleGuide)

It is encouraged that all TS and CSS code be run through formatters before
submitting a change. However, it is not a strict requirement enforced by CI.

### Installing npm Dependencies:

1. Install [docker](https://docs.docker.com/get-docker/)
2. Run `./npm install`

### Run ESLint, Stylelint, & Prettier

    ./npm run all
Hello World
