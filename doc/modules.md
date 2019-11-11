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

A [*module*](#glos-module) is a collection of packages that are released,
versioned, and distributed together. A module is identified by a [*module
path*](#glos-module-path), which is declared in a [`go.mod`
file](#go.mod-files), together with information about the module's
dependencies. The [*module root directory*](#glos-module-root-directory) is the
directory that contains the `go.mod` file. The [*main
module*](#glos-main-module) is the module containing the directory where the
`go` command is invoked.

Each [*package*](#glos-package) within a module is a collection of source files
in the same directory that are compiled together. A [*package
path*](#glos-package-path) is the module path joined with the subdirectory
containing the package (relative to the module root). For example, the module
`"golang.org/x/net"` contains a package in the directory `"html"`. That
package's path is `"golang.org/x/net/html"`.

<a id="versions"></a>
### Versions

A [*version*](#glos-version) identifies an immutable snapshot of a module, which
may be either a [release](#glos-release-version) or a
[pre-release](#glos-pre-release-version). Each version starts with the letter
`v`, followed by a semantic version. See [Semantic Versioning
2.0.0](https://semver.org/spec/v2.0.0.html) for details on how versions are
formatted, interpreted, and compared.

To summarize, a semantic version consists of three non-negative integers (the
major, minor, and patch versions, from left to right) separated by dots. The
patch version may be followed by an optional pre-release string starting with a
hyphen. The pre-release string or patch version may be followed by a build
metadata string starting with a plus. For example, `v0.0.0`, `v1.12.134`,
`v8.0.5-pre`, and `v2.0.9+meta` are valid versions.

Each part of a version indicates whether the version is stable and whether it is
compatible with previous versions.

* The [major version](#glos-major-version) must be incremented and the minor
  and patch versions must be set to zero after a backwards incompatible change
  is made to the module's public interface or documented functionality, for
  example, after a package is removed.
* The [minor version](#glos-minor-version) must be incremented and the patch
  version set to zero after a backwards compatible change, for example, after a
  new function is added.
* The [patch version](#glos-patch-version) must be incremented after a change
  that does not affect the module's public interface, such as a bug fix or
  optimization.
* The pre-release suffix indicates a version is a
  [pre-release](#glos-pre-release-version). Pre-release versions sort before
  the corresponding release versions. For example, `v1.2.3-pre` comes before
  `v1.2.3`.
* The build metadata suffix is ignored for the purpose of comparing versions.
  Tags with build metadata are ignored in version control repositories, but
  build metadata is preserved in versions specified in `go.mod` files. The
  suffix `+incompatible` denotes a version released before migrating to modules
  version major version 2 or later (see [Compatibility with non-module
  repositories](#non-module-compat).

A version is considered unstable if its major version is 0 or it has a
pre-release suffix. Unstable versions are not subject to compatibility
requirements. For example, `v0.2.0` may not be compatible with `v0.1.0`, and
`v1.5.0-beta` may not be compatible with `v1.5.0`.

Go may access modules in version control systems using tags, branches, or
revisions that don't follow these conventions. However, within the main module,
the `go` command will automatically convert revision names that don't follow
this standard into canonical versions. The `go` command will also remove build
metadata suffixes (except for `+incompatible`) as part of this process. This may
result in a [*pseudo-version*](#glos-pseudo-version), a pre-release version that
encodes a revision identifier (such as a Git commit hash) and a timestamp from a
version control system. For example, the command `go get -d
golang.org/x/net@daa7c041` will convert the commit hash `daa7c041` into the
pseudo-version `v0.0.0-20191109021931-daa7c04131f5`. Canonical versions are
required outside the main module, and the `go` command will report an error if a
non-canonical version like `master` appears in a `go.mod` file.

<a id="major-version-suffixes"></a>
### Major version suffixes

Starting with major version 2, module paths must have a [*major version
suffix*](#glos-major-version-suffix) like `/v2` that matches the major
version. For example, if a module has the path `example.com/mod` at `v1.0.0`, it
must have the path `example.com/mod/v2` at version `v2.0.0`.

Major version suffixes implement the [*import compatibility
rule*](https://research.swtch.com/vgo-import):

> If an old package and a new package have the same import path,
> the new package must be backwards compatible with the old package.

By definition, packages in a new major version of a module are not backwards
compatible with the corresponding packages in the previous major version.
Consequently, starting with `v2`, packages need new import paths. This is
accomplished by adding a major version suffix to the module path. Since the
module path is a prefix of the import path for each package within the module,
adding the major version suffix to the module path provides a distinct import
path for each incompatible version.

Major version suffixes are not allowed at major versions `v0` or `v1`. There is
no need to change the module path between `v0` and `v1` because `v0` versions
are unstable and have no compatibility guarantee. Additionally, for most
modules, `v1` is backwards compatible with the last `v0` version; a `v1` version
acts as a commitment to compatibility, rather than an indication of
incompatible changes compared with `v0`.

As a special case, modules paths starting with `gopkg.in/` must always have a
major version suffix, even at `v0` and `v1`. The suffix must start with a dot
rather than a slash (for example, `gopkg.in/yaml.v2`).

Major version suffixes let multiple major versions of a module coexist in the
same build. This may be necessary due to a [diamond dependency
problem](https://research.swtch.com/vgo-import#dependency_story). Ordinarily, if
a module is required at two different versions by transitive dependencies, the
higher version will be used. However, if the two versions are incompatible,
neither version will satisfy all clients. Since incompatible versions must have
different major version numbers, they must also have different module paths due
to major version suffixes. This resolves the conflict: modules with distinct
suffixes are treated as separate modules, and their packages—even packages in
same subdirectory relative to their module roots—are distinct.

Many Go projects released versions at `v2` or higher without using a major
version suffix before migrating to modules (perhaps before modules were even
introduced). These versions are annotated with a `+incompatible` build tag (for
example, `v2.0.0+incompatible`). See [Compatibility with non-module
repositories](#compatibility-with-non-module-repositories) for more information.

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
