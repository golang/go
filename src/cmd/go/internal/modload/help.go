// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import "cmd/go/internal/base"

var HelpModules = &base.Command{
	UsageLine: "modules",
	Short:     "modules, module versions, and more",
	Long: `
Modules are how Go manages dependencies.

A module is a collection of packages that are released, versioned, and
distributed together. Modules may be downloaded directly from version control
repositories or from module proxy servers.

For a series of tutorials on modules, see
https://golang.org/doc/tutorial/create-module.

For a detailed reference on modules, see https://golang.org/ref/mod.

Versioning is intrinsic to go modules, it should be noted that once the
version of a public module has been defined and registered with a public
proxy server; The version is permanent and can no longer be modified.

For a detailed reference on versioning, see
https://golang.org/doc/modules/version-numbers

By default, the go command may download modules from https://proxy.golang.org.
It may authenticate modules using the checksum database at
https://sum.golang.org. Both services are operated by the Go team at Google.
The privacy policies for these services are available at
https://proxy.golang.org/privacy and https://sum.golang.org/privacy,
respectively.

The go command's download behavior may be configured using GOPROXY, GOSUMDB,
GOPRIVATE, and other environment variables. See 'go help environment'
and https://golang.org/ref/mod#private-module-privacy for more information.

Public modules may incur delays upon modification, thus new versions may not
be available for remote update instantaneously.

In the case that a rapid cycle of continuous integration be required; The
module can either be made accessible locally, by way of the 'replace'
directive, else, the public version proxy cache can be accelerated by
revving the modules version number and the use of an in development
numbering strategy.
	`,
}

var HelpGoMod = &base.Command{
	UsageLine: "go.mod",
	Short:     "the go.mod file",
	Long: `
A module version is defined by a tree of source files, with a go.mod
file in its root. When the go command is run, it looks in the current
directory and then successive parent directories to find the go.mod
marking the root of the main (current) module.

The go.mod file format is described in detail at
https://golang.org/ref/mod#go-mod-file.

To create a new go.mod file, use 'go mod init'. For details see
'go help mod init' or https://golang.org/ref/mod#go-mod-init.

To add missing module requirements or remove unneeded requirements,
use 'go mod tidy'. For details, see 'go help mod tidy' or
https://golang.org/ref/mod#go-mod-tidy.

To add, upgrade, downgrade, or remove a specific module requirement, use
'go get'. For details, see 'go help module-get' or
https://golang.org/ref/mod#go-get.

To make other changes or to parse go.mod as JSON for use by other tools,
use 'go mod edit'. See 'go help mod edit' or
https://golang.org/ref/mod#go-mod-edit.
	`,
}
