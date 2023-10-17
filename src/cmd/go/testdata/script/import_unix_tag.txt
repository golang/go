# Regression test for https://go.dev/issue/54712: the "unix" build constraint
# was not applied consistently during package loading.

go list -x -f '{{if .Module}}{{.ImportPath}}{{end}}' -deps .
stdout 'example.com/version'

-- go.mod --
module example

go 1.19

require example.com/version v1.1.0
-- go.sum --
example.com/version v1.1.0 h1:VdPnGmIF1NJrntStkxGrF3L/OfhaL567VzCjncGUgtM=
example.com/version v1.1.0/go.mod h1:S7K9BnT4o5wT4PCczXPfWVzpjD4ud4e7AJMQJEgiu2Q=
-- main_notunix.go --
//go:build !(aix || darwin || dragonfly || freebsd || hurd || linux || netbsd || openbsd || solaris)

package main

import _ "example.com/version"

func main() {}

-- main_unix.go --
//go:build unix

package main

import _ "example.com/version"

func main() {}
