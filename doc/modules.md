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
### `GOPROXY` protocol

A [*module proxy*](#glos-module-proxy) is an HTTP server that can respond to
`GET` requests for paths specified below. The requests have no query parameters,
and no specific headers are required, so even a site serving from a fixed file
system (including a `file://` URL) can be a module proxy.

Successful HTTP responses must have the status code 200 (OK). Redirects (3xx)
are followed. Responses with status codes 4xx and 5xx are treated as errors.
The error codes 404 (Not Found) and 410 (Gone) indicate that the
requested module or version is not available on the proxy, but it may be found
elsewhere. Error responses should have content type `text/plain` with
`charset` either `utf-8` or `us-ascii`.

The `go` command may be configured to contact proxies or source control servers
using the `GOPROXY` environment variable, which is a comma-separated list of
URLs or the keywords `direct` or `off` (see [Environment
variables](#environment-variables) for details). When the `go` command receives
a 404 or 410 response from a proxy, it falls back to later proxies in the
list. The `go` command does not fall back to later proxies in response to other
4xx and 5xx errors. This allows a proxy to act as a gatekeeper, for example, by
responding with error 403 (Forbidden) for modules not on an approved list.

The table below specifies queries that a module proxy must respond to. For each
path, `$base` is the path portion of a proxy URL,`$module` is a module path, and
`$version` is a version. For example, if the proxy URL is
`https://example.com/mod`, and the client is requesting the `go.mod` file for
the module `golang.org/x/text` at version `v0.3.2`, the client would send a
`GET` request for `https://example.com/mod/golang.org/x/text/@v/v0.3.2.mod`.

To avoid ambiguity when serving from case-insensitive file systems,
the `$module` and `$version` elements are case-encoded by replacing every
uppercase letter with an exclamation mark followed by the corresponding
lower-case letter. This allows modules `example.com/M` and `example.com/m` to
both be stored on disk, since the former is encoded as `example.com/!m`.

<!-- TODO(jayconrod): This table has multi-line cells, and GitHub Flavored
Markdown doesn't have syntax for that, so we use raw HTML. Gitiles doesn't
include this table in the rendered HTML. Once x/website has a Markdown renderer,
ensure this table is readable. If the cells are too large, and it's difficult
to scan, use paragraphs or sections below.
-->

<table>
  <thead>
    <tr>
      <th>Path</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>$base/$module/@v/list</code></td>
      <td>
        Returns a list of known versions of the given module in plain text, one
        per line. This list should not include pseudo-versions.
      </td>
    </tr>
    <tr>
      <td><code>$base/$module/@v/$version.info</code></td>
      <td>
        <p>
          Returns JSON-formatted metadata about a specific version of a module.
          The response must be a JSON object that corresponds to the Go data
          structure below:
        </p>
        <pre>
type Info struct {
    Version string    // version string
    Time    time.Time // commit time
}
        </pre>
        <p>
          The <code>Version</code> field is required and must contain a valid,
          <a href="#glos-canonical-version">canonical version</a> (see
          <a href="#versions">Versions</a>). The <code>$version</code> in the
          request path does not need to be the same version or even a valid
          version; this endpoint may be used to find versions for branch names
          or revision identifiers. However, if <code>$version</code> is a
          canonical version with a major version compatible with
          <code>$module</code>, the <code>Version</code> field in a successful
          response must be the same.
        </p>
        <p>
          The <code>Time</code> field is optional. If present, it must be a
          string in RFC 3339 format. It indicates the time when the version
          was created.
        </p>
        <p>
          More fields may be added in the future, so other names are reserved.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>$base/$module/@v/$version.mod</code></td>
      <td>
        Returns the <code>go.mod</code> file for a specific version of a
        module. If the module does not have a <code>go.mod</code> file at the
        requested version, a file containing only a <code>module</code>
        statement with the requested module path must be returned. Otherwise,
        the original, unmodified <code>go.mod</code> file must be returned.
      </td>
    </tr>
    <tr>
      <td><code>$base/$module/@v/$version.zip</code></td>
      <td>
        Returns a zip file containing the contents of a specific version of
        a module. See <a href="#zip-format">Module zip format</a> for details
        on how this zip file must be formatted.
      </td>
    </tr>
    <tr>
      <td><code>$base/$module/@latest</code></td>
      <td>
        Returns JSON-formatted metadata about the latest known version of a
        module in the same format as
        <code>$base/$module/@v/$version.info</code>. The latest version should
        be the version of the module that the <code>go</code> command should use
        if <code>$base/$module/@v/list</code> is empty or no listed version is
        suitable. This endpoint is optional, and module proxies are not required
        to implement it.
      </td>
    </tr>
  </tbody>
</table>

When resolving the latest version of a module, the `go` command will request
`$base/$module/@v/list`, then, if no suitable versions are found,
`$base/$module/@latest`. The `go` command prefers, in order: the semantically
highest release version, the semantically highest pre-release version, and the
chronologically most recent pseudo-version. In Go 1.12 and earlier, the `go`
command considered pseudo-versions in `$base/$module/@v/list` to be pre-release
versions, but this is no longer true since Go 1.13.

A module proxy must always serve the same content for successful
responses for `$base/$module/$version.mod` and `$base/$module/$version.zip`
queries. This content is [cryptographically authenticated](#authenticating)
using [`go.sum` files](#go.sum-file-format) and, by default, the
[checksum database](#checksum-database).

The `go` command caches most content it downloads from module proxies in its
module cache in `$GOPATH/pkg/mod/cache/download`. Even when downloading directly
from version control systems, the `go` command synthesizes explicit `info`,
`mod`, and `zip` files and stores them in this directory, the same as if it had
downloaded them directly from a proxy. The cache layout is the same as the proxy
URL space, so serving `$GOPATH/pkg/mod/cache/download` at (or copying it to)
`https://example.com/proxy` would let users access cached module versions by
setting `GOPROXY` to `https://example.com/proxy`.

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

<a id="glos-canonical-version">
**canonical version:** A correctly formatted [version](#glos-version) without
a build metadata suffix other than `+incompatible`. For example, `v1.2.3`
is a canonical version, but `v1.2.3+meta` is not.

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

<a id="glos-module-proxy"></a>
**module proxy:** A web server that implements the [`GOPROXY`
protocol](#goproxy-protocol). The `go` command downloads version information,
`go.mod` files, and module zip files from module proxies.

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
