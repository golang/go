<!--{
  "Title": "Go Modules Reference",
  "Subtitle": "Version of Sep 4, 2019",
  "Path": "/ref/modules"
}-->
<!-- TODO(jayconrod): ensure golang.org/x/website can render Markdown or convert
this document to HTML before Go 1.14. -->
<!-- TODO(jayconrod): ensure anchors work correctly after Markdown rendering -->

<a id="introduction"></a>
## Introduction

<a id="modules-overview"></a>
## Modules, packages, and versions

<a id="versions"></a>
### Versions

<a id="major-version-suffixes"></a>
### Major version suffixes

<a id="resolve-pkg-mod"></a>
### Resolving a package to a module

<a id="go.mod-files"></a>
## `go.mod` files

<a id="go.mod-file-format"></a>
### `go.mod` file format

<a id="minimal-version-selection"></a>
### Minimal version selection (MVS)

<a id="non-module-compat"></a>
### Compatibility with non-module repositories

<a id="mod-commands"></a>
## Module-aware build commands

<a id="enabling"></a>
### Enabling modules

<a id="initializing"></a>
### Initializing modules

<a id="build-commands"></a>
### Build commands

<a id="vendoring"></a>
### Vendoring

<a id="go-mod-download"></a>
### `go mod download`

<a id="go-mod-verify"></a>
### `go mod verify`

<a id="go-mod-edit"></a>
### `go mod edit`

<a id="go-clean-modcache"></a>
### `go clean -modcache`

<a id="commands-outside"></a>
### Module commands outside a module

<a id="retrieving-modules"></a>
## Retrieving modules

<a id="goproxy-protocol"></a>
### GOPROXY protocol

<a id="communicating-with-proxies"></a>
### Communicating with proxies

<a id="communicating-with-vcs"></a>
### Communicating with version control systems

<a id="custom-import-paths"></a>
### Custom import paths

<!-- TODO(jayconrod): custom import paths, details of direct mode -->

<a id="path-constraints"></a>
### File name and path constraints

<a id="zip-format"></a>
### Module zip format

<a id="private-modules"></a>
### Private modules

<a id="authenticating"></a>
## Authenticating modules

<a id="go.sum-file-format"></a>
### go.sum file format

<a id="checksum-database"></a>
### Checksum database

<a id="privacy"></a>
## Privacy

<a id="environment-variables"></a>
## Environment variables

<a id="glossary">
## Glossary

<a id="glos-build-list"></a>
**build list:** The list of module versions that will be used for a build
command such as `go build`, `go list`, or `go test`. The build list is
determined from the [main module's](#glos-main-module) [`go.mod`
file](#glos-go.mod-file) and `go.mod` files in transitively required modules
using [minimal version selection](#glos-minimal-version-selection). The build
list contains versions for all modules in the [module
graph](#glos-module-graph), not just those relevant to a specific command.

<a id="glos-go.mod-file"></a>
**`go.mod` file:** The file that defines a module's path, requirements, and
other metadata. Appears in the [module's root
directory](#glos-module-root-directory). See the section on [`go.mod`
files](#go.mod-files).

<a id="glos-import-path"></a>
**import path:** A string used to import a package in a Go source file.
Synonymous with [package path](#glos-package-path).

<a id="glos-main-module"></a>
**main module:** The module in which the `go` command is invoked.

<a id="glos-major-version"></a>
**major version:** The first number in a semantic version (`1` in `v1.2.3`). In
a release with incompatible changes, the major version must be incremented, and
the minor and patch versions must be set to 0. Semantic versions with major
version 0 are considered unstable.

<a id="glos-major-version-suffix"></a>
**major version suffix:** A module path suffix that matches the major version
number. For example, `/v2` in `example.com/mod/v2`. Major version suffixes are
required at `v2.0.0` and later and are not allowed at earlier versions. See
the section on [Major version suffixes](#major-version-suffixes).

<a id="glos-minimal-version-selection"></a>
**minimal version selection (MVS):** The algorithm used to determine the
versions of all modules that will be used in a build. See the section on
[Minimal version selection](#minimal-version-selection) for details.

<a id="glos-minor-version"></a>
**minor version:** The second number in a semantic version (`2` in `v1.2.3`). In
a release with new, backwards compatible functionality, the minor version must
be incremented, and the patch version must be set to 0.

<a id="glos-module"></a>
**module:** A collection of packages that are released, versioned, and
distributed together.

<a id="glos-module-graph"></a>
**module graph:** The directed graph of module requirements, rooted at the [main
module](#glos-main-module). Each vertex in the graph is a module; each edge is a
version from a `require` statement in a `go.mod` file (subject to `replace` and
`exclude` statements in the main module's `go.mod` file.

<a id="glos-module-path"></a>
**module path:** A path that identifies a module and acts as a prefix for
package import paths within the module. For example, `"golang.org/x/net"`.

<a id="glos-module-root-directory"></a>
**module root directory:** The directory that contains the `go.mod` file that
defines a module.

<a id="glos-package"></a>
**package:** A collection of source files in the same directory that are
compiled together. See the [Packages section](/ref/spec#Packages) in the Go
Language Specification.

<a id="glos-package-path"></a>
**package path:** The path that uniquely identifies a package. A package path is
a [module path](#glos-module-path) joined with a subdirectory within the module.
For example `"golang.org/x/net/html"` is the package path for the package in the
module `"golang.org/x/net"` in the `"html"` subdirectory. Synonym of
[import path](#glos-import-path).

<a id="glos-patch-version"></a>
**patch version:** The third number in a semantic version (`3` in `v1.2.3`). In
a release with no changes to the module's public interface, the patch version
must be incremented.

<a id="glos-pre-release-version"></a>
**pre-release version:** A version with a dash followed by a series of
dot-separated identifiers immediately following the patch version, for example,
`v1.2.3-beta4`. Pre-release versions are considered unstable and are not
assumed to be compatible with other versions. A pre-release version sorts before
the corresponding release version: `v1.2.3-pre` comes before `v1.2.3`. See also
[release version](#glos-release-version).

<a id="glos-pseudo-version"></a>
**pseudo-version:** A version that encodes a revision identifier (such as a Git
commit hash) and a timestamp from a version control system. For example,
`v0.0.0-20191109021931-daa7c04131f5`. Used for [compatibility with non-module
repositories](#non-module-compat) and in other situations when a tagged
version is not available.

<a id="glos-release-version"></a>
**release version:** A version without a pre-release suffix. For example,
`v1.2.3`, not `v1.2.3-pre`. See also [pre-release
version](#glos-pre-release-version).

<a id="glos-version"></a>
**version:** An identifier for an immutable snapshot of a module, written as the
letter `v` followed by a semantic version. See the section on
[Versions](#versions).
