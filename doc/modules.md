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

<a id="module-path"></a>
### Module paths

A [*module path*](#glos-module-path) is the canonical name for a module,
declared with the [`module` directive](#go.mod-module) in the module's
[`go.mod` file](#glos-go.mod-file). A module's path is the prefix for package
paths within the module.

A module path should describe both what the module does and where to find it.
Typically, a module path consists of a repository root path, a subdirectory
within the repository (usually empty), and a major version suffix (for major
version 2 or higher).

* The repository root path is part of a URL that corresponds to a version
  control repository. For example, `golang.org/x/net`. See [Custom import
  paths](#custom-import-paths) for details on how paths are resolved to
  repositories.
* If the module is in a subdirectory of the version control repository, the
  subdirectory should be part of the module path (but not the repository root
  path). For example, the module `golang.org/x/tools/gopls` is in the `/gopls`
  subdirectory of the repository `golang.org/x/tools`.
* If the module is released at major version 2 or higher, the module path must
  end with a [major version suffix](#major-version-suffixes) like
  `/v2`. This may or may not be part of the subdirectory name. For example, the
  module with path `golang.org/x/repo/sub/v2` could be in the `/sub` or
  `/sub/v2` subdirectory of the repository `golang.org/x/repo`.

If a module might be depended on by other modules, these rules must be followed
so that the `go` command can find and download the module. There are also
several [lexical restrictions](#go.mod-ident) on characters allowed in
module paths.

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
repositories](#non-module-compat) for more information.

<a id="resolve-pkg-mod"></a>
### Resolving a package to a module

When the `go` command loads a package using a [package
path](#glos-package-path), it needs to determine which module provides the
package.

The `go` command starts by searching the [build list](#glos-build-list) for
modules with paths that are prefixes of the package path. For example, if the
package `example.com/a/b` is imported, and the module `example.com/a` is in the
build list, the `go` command will check whether `example.com/a` contains the
package, in the directory `b`. At least one file with the `.go` extension must
be present in a directory for it to be considered a package. [Build
constraints](/pkg/go/build/#hdr-Build_Constraints) are not applied for this
purpose. If exactly one module in the build list provides the package, that
module is used. If two or more modules provide the package, an error is
reported. If no modules provide the package, the `go` command will attempt to
find a new module (unless the flags `-mod=readonly` or `-mod=vendor` are used,
in which case, an error is reported).

<!-- NOTE(golang.org/issue/27899): the go command reports an error when two
or more modules provide a package with the same path as above. In the future,
we may try to upgrade one (or all) of the colliding modules.
-->

When the `go` command looks up a new module for a package path, it checks the
`GOPROXY` environment variable, which is a comma-separated list of proxy URLs or
the keywords `direct` or `off`. A proxy URL indicates the `go` command should
contact a [module proxy](#glos-module-proxy) using the [`GOPROXY`
protocol](#goproxy-protocol). `direct` indicates that the `go` command should
[communicate with a version control system](#communicating-with-vcs). `off`
indicates that no communication should be attempted. The `GOPRIVATE` and
`GONOPROXY` [environment variables](#environment-variables) can also be used to
control this behavior.

For each entry in the `GOPROXY` list, the `go` command requests the latest
version of each module path that might provide the package (that is, each prefix
of the package path). For each successfully requested module path, the `go`
command will download the module at the latest version and check whether the
module contains the requested package. If one or more modules contain the
requested package, the module with the longest path is used. If one or more
modules are found but none contain the requested package, an error is
reported. If no modules are found, the `go` command tries the next entry in the
`GOPROXY` list. If no entries are left, an error is reported.

For example, suppose the `go` command is looking for a module that provides the
package `golang.org/x/net/html`, and `GOPROXY` is set to
`https://corp.example.com,https://proxy.golang.org`. The `go` command may make
the following requests:

* To `https://corp.example.com/` (in parallel):
  * Request for latest version of `golang.org/x/net/html`
  * Request for latest version of `golang.org/x/net`
  * Request for latest version of `golang.org/x`
  * Request for latest version of `golang.org`
* To `https://proxy.golang.org/`, if all requests to `https://corp.example.com/`
  have failed with 404 or 410:
  * Request for latest version of `golang.org/x/net/html`
  * Request for latest version of `golang.org/x/net`
  * Request for latest version of `golang.org/x`
  * Request for latest version of `golang.org`

After a suitable module has been found, the `go` command will add a new
[requirement](#go.mod-require) with the new module's path and version to the
main module's `go.mod` file. This ensures that when the same package is loaded
in the future, the same module will be used at the same version. If the resolved
package is not imported by a package in the main module, the new requirement
will have an `// indirect` comment.

<a id="go.mod-files"></a>
## `go.mod` files

A module is defined by a UTF-8 encoded text file named `go.mod` in its root
directory. The `go.mod` file is line-oriented. Each line holds a single
directive, made up of a keyword followed by arguments. For example:

```
module example.com/my/thing

go 1.12

require example.com/other/thing v1.0.2
require example.com/new/thing/v2 v2.3.4
exclude example.com/old/thing v1.2.3
replace example.com/bad/thing v1.4.5 => example.com/good/thing v1.4.5
```

The leading keyword can be factored out of adjacent lines to create a block,
like in Go imports.

```
require (
    example.com/new/thing/v2 v2.3.4
    example.com/old/thing v1.2.3
)
```

The `go.mod` file is designed to be human readable and machine writable. The
`go` command provides several subcommands that change `go.mod` files. For
example, [`go get`](#go-get) can upgrade or downgrade specific dependencies.
Commands that load the module graph will [automatically update](#go.mod-updates)
`go.mod` when needed. [`go mod edit`](#go-mod-tidy) can perform low-level edits.
The
[`golang.org/x/mod/modfile`](https://pkg.go.dev/golang.org/x/mod/modfile?tab=doc)
package can be used by Go programs to make the same changes programmatically.

<a id="go.mod-lexical"></a>
### Lexical elements

When a `go.mod` file is parsed, its content is broken into a sequence of tokens.
There are several kinds of tokens: whitespace, comments, punctuation,
keywords, identifiers, and strings.

*White space* consists of spaces (U+0020), tabs (U+0009), carriage returns
(U+000D), and newlines (U+000A). White space characters other than newlines have
no effect except to separate tokens that would otherwise be combined. Newlines
are significant tokens.

*Comments* start with `//` and run to the end of a line. `/* */` comments are
not allowed.

*Punctuation* tokens include `(`, `)`, and `=>`.

*Keywords* distinguish different kinds of directives in a `go.mod` file. Allowed
keywords are `module`, `go`, `require`, `replace`, and `exclude`.

*Identifiers* are sequences of non-whitespace characters, such as module paths
or semantic versions.

*Strings* are quoted sequences of characters. There are two kinds of strings:
interpreted strings beginning and ending with quotation marks (`"`, U+0022) and
raw strings beginning and ending with grave accents (<code>&#60;</code>,
U+0060). Interpreted strings may contain escape sequences consisting of a
backslash (`\`, U+005C) followed by another character. An escaped quotation
mark (`\"`) does not terminate an interpreted string. The unquoted value
of an interpreted string is the sequence of characters between quotation
marks with each escape sequence replaced by the character following the
backslash (for example, `\"` is replaced by `"`, `\n` is replaced by `n`).
In contrast, the unquoted value of a raw string is simply the sequence of
characters between grave accents; backslashes have no special meaning within
raw strings.

Identifiers and strings are interchangeable in the `go.mod` grammar.

<a id="go.mod-ident"></a>
### Module paths and versions

Most identifiers and strings in a `go.mod` file are either module paths or
versions.

A module path must satisfy the following requirements:

* The path must consist of one or more path elements separated by slashes
  (`/`, U+002F). It must not begin or end with a slash.
* Each path element is a non-empty string made of up ASCII letters, ASCII
  digits, and limited ASCII punctuation (`+`, `-`, `.`, `_`, and `~`).
* A path element may not begin or end with a dot (`.`, U+002E).
* The element prefix up to the first dot must not be a reserved file name on
  Windows, regardless of case (`CON`, `com1`, `NuL`, and so on).

If the module path appears in a `require` directive and is not replaced, or
if the module paths appears on the right side of a `replace` directive,
the `go` command may need to download modules with that path, and some
additional requirements must be satisfied.

* The leading path element (up to the first slash, if any), by convention a
  domain name, must contain only lower-case ASCII letters, ASCII digits, dots
  (`.`, U+002E), and dashes (`-`, U+002D); it must contain at least one dot and
  cannot start with a dash.
* For a final path element of the form `/vN` where `N` looks numeric (ASCII
  digits and dots), `N` must not begin with a leading zero, must not be `/v1`,
  and must not contain any dots.
  * For paths beginning with `gopkg.in/`, this requirement is replaced by a
    requirement that the path follow the [gopkg.in](https://gopkg.in) service's
    conventions.

Versions in `go.mod` files may be [canonical](#glos-canonical-version) or
non-canonical.

A canonical version starts with the letter `v`, followed by a semantic version
following the [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
specification. See [Versions](#versions) for more information.

Most other identifiers and strings may be used as non-canonical versions, though
there are some restrictions to avoid problems with file systems, repositories,
and [module proxies](#glos-module-proxy). Non-canonical versions are only
allowed in the main module's `go.mod` file. The `go` command will attempt to
replace each non-canonical version with an equivalent canonical version when it
automatically [updates](#go.mod-updates) the `go.mod` file.

In places where a module path is associated with a verison (as in `require`,
`replace`, and `exclude` directives), the final path element must be consistent
with the version. See [Major version suffixes](#major-version-suffixes).

<a id="go.mod-grammar"></a>
### Grammar

`go.mod` syntax is specified below using Extended Backus-Naur Form (EBNF).
See the [Notation section in the Go Language Specificiation](/ref/spec#Notation)
for details on EBNF syntax.

```
GoMod = { Directive } .
Directive = ModuleDirective |
            GoDirective |
            RequireDirective |
            ExcludeDirective |
            ReplaceDirective .
```

Newlines, identifiers, and strings are denoted with `newline`, `ident`, and
`string`, respectively.

Module paths and versions are denoted with `ModulePath` and `Version`.

```
ModulePath = ident | string . /* see restrictions above */
Version = ident | string .    /* see restrictions above */
```

<a id="go.mod-module"></a>
### `module` directive

A `module` directive defines the main module's [path](#glos-module-path). A
`go.mod` file must contain exactly one `module` directive.

```
ModuleDirective = "module" ( ModulePath | "(" newline ModulePath newline ")" newline .
```

Example:

```
module golang.org/x/net
```

<a id="go.mod-go"></a>
### `go` directive

A `go` directive sets the expected language version for the module. The
version must be a valid Go release version: a positive integer followed by a dot
and a non-negative integer (for example, `1.9`, `1.14`).

The language version determines which language features are available when
compiling packages in the module. Language features present in that version
will be available for use. Language features removed in earlier versions,
or added in later versions, will not be available. The language version does not
affect build tags, which are determined by the Go release being used.

The language version is also used to enable features in the `go` command. For
example, automatic [vendoring](#vendoring) may be enabled with a `go` version of
`1.14` or higher.

A `go.mod` file may contain at most one `go` directive. Most commands will add a
`go` directive with the current Go version if one is not present.

```
GoDirective = "go" GoVersion newline .
GoVersion = string | ident .  /* valid release version; see above */
```

Example:

```
go 1.14
```

<a id="go.mod-require"></a>
### `require` directive

A `require` directive declares a minimum required version of a given module
dependency. For each required module version, the `go` command loads the
`go.mod` file for that version and incorporates the requirements from that
file. Once all requirements have been loaded, the `go` command resolves them
using [minimal version selection (MVS)](#minimal-version-selection) to produce
the [build list](#glos-build-list).

The `go` command automatically adds `// indirect` comments for some
requirements. An `// indirect` comment indicates that no package from the
required module is directly imported by any package in the main module.
The `go` command adds an indirect requirement when the selected version of a
module is higher than what is already implied (transitively) by the main
module's other dependencies. That may occur because of an explicit upgrade
(`go get -u`), removal of some other dependency that previously imposed the
requirement (`go mod tidy`), or a dependency that imports a package without
a corresponding requirement in its own `go.mod` file (such as a dependency
that lacks a `go.mod` file altogether).

```
RequireDirective = "require" ( RequireSpec | "(" newline { RequireSpec } ")" newline ) .
RequireSpec = ModulePath Version newline .
```

Example:

```
require golang.org/x/net v1.2.3

require (
    golang.org/x/crypto v1.4.5 // indirect
    golang.org/x/text v1.6.7
)
```

<a id="go.mod-exclude"></a>
### `exclude` directive

An `exclude` directive prevents a module version from being loaded by the `go`
command. If an excluded version is referenced by a `require` directive in a
`go.mod` file, the `go` command will list available versions for the module (as
shown with `go list -m -versions`) and will load the next higher non-excluded
version instead. Both release and pre-release versions are considered for this
purpose, but pseudo-versions are not. If there are no higher versions,
the `go` command will report an error. Note that this [may
change](https://golang.org/issue/36465) in Go 1.15.

<!-- TODO(golang.org/issue/36465): update after change -->

`exclude` directives only apply in the main module's `go.mod` file and are
ignored in other modules. See [Minimal version
selection](#minimal-version-selection) for details.

```
ExcludeDirective = "exclude" ( ExcludeSpec | "(" newline { ExcludeSpec } ")" ) .
ExcludeSpec = ModulePath Version newline .
```

Example:

```
exclude golang.org/x/net v1.2.3

exclude (
    golang.org/x/crypto v1.4.5
    golang.org/x/text v1.6.7
)
```

<a id="go.mod-replace"></a>
### `replace` directive

A `replace` directive replaces the contents of a specific version of a module,
or all versions of a module, with contents found elsewhere. The replacement
may be specified with either another module path and version, or a
platform-specific file path.

If a version is present on the left side of the arrow (`=>`), only that specific
version of the module is replaced; other versions will be accessed normally.
If the left version is omitted, all versions of the module are replaced.

If the path on the right side of the arrow is an absolute or relative path
(beginning with `./` or `../`), it is interpreted as the local file path to the
replacement module root directory, which must contain a `go.mod` file. The
replacement version must be omitted in this case.

If the path on the right side is not a local path, it must be a valid module
path. In this case, a version is required. The same module version must not
also appear in the build list.

Regardless of whether a replacement is specified with a local path or module
path, if the replacement module has a `go.mod` file, its `module` directive
must match the module path it replaces.

`replace` directives only apply in the main module's `go.mod` file
and are ignored in other modules. See [Minimal version
selection](#minimal-version-selection) for details.

```
ReplaceDirective = "replace" ( ReplaceSpec | "(" newline { ReplaceSpec } ")" newline ")" ) .
ReplaceSpec = ModulePath [ Version ] "=>" FilePath newline
            | ModulePath [ Version ] "=>" ModulePath Version newline .
FilePath = /* platform-specific relative or absolute file path */
```

Example:

```
replace golang.org/x/net v1.2.3 => example.com/fork/net v1.4.5

replace (
    golang.org/x/net v1.2.3 => example.com/fork/net v1.4.5
    golang.org/x/net => example.com/fork/net v1.4.5
    golang.org/x/net v1.2.3 => ./fork/net
    golang.org/x/net => ./fork/net
)
```

<a id="go.mod-updates"></a>
### Automatic updates

The `go` command automatically updates `go.mod` when it uses the module graph if
some information is missing or `go.mod` doesn't accurately reflect reality.  For
example, consider this `go.mod` file:

```
module example.com/M

require (
    example.com/A v1
    example.com/B v1.0.0
    example.com/C v1.0.0
    example.com/D v1.2.3
    example.com/E dev
)

exclude example.com/D v1.2.3
```

The update rewrites non-canonical version identifiers to
[canonical](#glos-canonical-version) semver form, so `example.com/A`'s `v1`
becomes `v1.0.0`, and `example.com/E`'s `dev` becomes the pseudo-version for the
latest commit on the `dev` branch, perhaps `v0.0.0-20180523231146-b3f5c0f6e5f1`.

The update modifies requirements to respect exclusions, so the requirement on
the excluded `example.com/D v1.2.3` is updated to use the next available version
of `example.com/D`, perhaps `v1.2.4` or `v1.3.0`.

The update removes redundant or misleading requirements. For example, if
`example.com/A v1.0.0` itself requires `example.com/B v1.2.0` and `example.com/C
v1.0.0`, then `go.mod`'s requirement of `example.com/B v1.0.0` is misleading
(superseded by `example.com/A`'s need for `v1.2.0`), and its requirement of
`example.com/C v1.0.0` is redundant (implied by `example.com/A`'s need for the
same version), so both will be removed. If the main module contains packages
that directly import packages from `example.com/B` or `example.com/C`, then the
requirements will be kept but updated to the actual versions being used.

Finally, the update reformats the `go.mod` in a canonical formatting, so
that future mechanical changes will result in minimal diffs. The `go` command
will not update `go.mod` if only formatting changes are needed.

Because the module graph defines the meaning of import statements, any commands
that load packages also use and therefore update `go.mod`, including `go build`,
`go get`, `go install`, `go list`, `go test`, `go mod graph`, `go mod tidy`, and
`go mod why`.

The `-mod=readonly` flag prevents commands from automatically updating
`go.mod`. However, if a command needs to perform an action that would
update to `go.mod`, it will report an error. For example, if
`go build` is asked to build a package not provided by any module in the build
list, `go build` will report an error instead of looking up the module and
updating requirements in `go.mod`.

<a id="minimal-version-selection"></a>
## Minimal version selection (MVS)

<a id="non-module-compat"></a>
## Compatibility with non-module repositories

<a id="mod-commands"></a>
## Module-aware commands

Most `go` commands may run in *Module-aware mode* or *`GOPATH` mode*. In
module-aware mode, the `go` command uses `go.mod` files to find versioned
dependencies, and it typically loads packages out of the [module
cache](#glos-module-cache), downloading modules if they are missing. In `GOPATH`
mode, the `go` command ignores modules; it looks in `vendor` directories and in
`GOPATH` to find dependencies.

Module-aware mode is active by default whenever a `go.mod` file is found in the
current directory or in any parent directory. For more fine-grained control, the
`GO111MODULE` environment variable may be set to one of three values: `on`,
`off`, or `auto`.

* If `GO111MODULE=off`, the `go` command ignores `go.mod` files and runs in
  `GOPATH` mode.
* If `GO111MODULE=on`, the `go` command runs in module-aware mode, even when
  no `go.mod` file is present. Not all commands work without a `go.mod` file:
  see [Module commands outside a module](#commands-outside).
* If `GO111MODULE=auto` or is unset, the `go` command runs in module-aware
  mode if a `go.mod` file is present in the current directory or any parent
  directory (the default behavior).

In module-aware mode, `GOPATH` no longer defines the meaning of imports during a
build, but it still stores downloaded dependencies (in `GOPATH/pkg/mod`; see
[Module cache](#module-cache)) and installed commands (in `GOPATH/bin`, unless
`GOBIN` is set).

<a id="build-commands"></a>
### Build commands

<a id="vendoring"></a>
### Vendoring

<a id="go-get"></a>
### `go get`

<a id="go-list-m"></a>
### `go list -m`

Usage:

```
go list -m [-u] [-versions] [list flags] [modules]
```

Example:

```
$ go list -m all
$ go list -m -versions example.com/m
$ go list -m -json example.com/m@latest
```

The `-m` flag causes `go list` to list modules instead of packages. In this
mode, the arguments to `go list` may be modules, module patterns (containing the
`...` wildcard), [module queries](#module-queries), or the special pattern
`all`, which matches all modules in the [build list](#glos-build-list). If no
arguments are specified, the [main module](#glos-main-module) is listed.

When listing modules, the `-f` flag still specifies a format template applied
to a Go struct, but now a `Module` struct:

```
type Module struct {
    Path      string       // module path
    Version   string       // module version
    Versions  []string     // available module versions (with -versions)
    Replace   *Module      // replaced by this module
    Time      *time.Time   // time version was created
    Update    *Module      // available update, if any (with -u)
    Main      bool         // is this the main module?
    Indirect  bool         // is this module only an indirect dependency of main module?
    Dir       string       // directory holding files for this module, if any
    GoMod     string       // path to go.mod file for this module, if any
    GoVersion string       // go version used in module
    Error     *ModuleError // error loading module
}

type ModuleError struct {
    Err string // the error itself
}
```

The default output is to print the module path and then information about the
version and replacement if any. For example, `go list -m all` might print:

```
example.com/main/module
golang.org/x/text v0.3.0 => /tmp/text
rsc.io/pdf v0.1.1
```

The `Module` struct has a `String` method that formats this line of output, so
that the default format is equivalent to `-f '{{.String}}'`.

Note that when a module has been replaced, its `Replace` field describes the
replacement module module, and its `Dir` field is set to the replacement
module's source code, if present. (That is, if `Replace` is non-nil, then `Dir`
is set to `Replace.Dir`, with no access to the replaced source code.)

The `-u` flag adds information about available upgrades. When the latest version
of a given module is newer than the current one, `list -u` sets the module's
`Update` field to information about the newer module. The module's `String`
method indicates an available upgrade by formatting the newer version in
brackets after the current version. For example, `go list -m -u all` might
print:

```
example.com/main/module
golang.org/x/text v0.3.0 [v0.4.0] => /tmp/text
rsc.io/pdf v0.1.1 [v0.1.2]
```

(For tools, `go list -m -u -json all` may be more convenient to parse.)

The `-versions` flag causes `list` to set the module's `Versions` field to a
list of all known versions of that module, ordered according to semantic
versioning, lowest to highest. The flag also changes the default output format
to display the module path followed by the space-separated version list.

The template function `module` takes a single string argument that must be a
module path or query and returns the specified module as a `Module` struct. If
an error occurs, the result will be a `Module` struct with a non-nil `Error`
field.

<a id="go-mod-download"></a>
### `go mod download`

Usage:

```
go mod download [-json] [-x] [modules]
```

Example:

```
$ go mod download
$ go mod download golang.org/x/mod@v0.2.0
```

The `go mod download` command downloads the named modules into the [module
cache](#glos-module-cache). Arguments can be module paths or module
patterns selecting dependencies of the main module or [module
queries](#module-queries) of the form `path@version`. With no arguments,
`download` applies to all dependencies of the [main module](#glos-main-module).

The `go` command will automatically download modules as needed during ordinary
execution. The `go mod download` command is useful mainly for pre-filling the
module cache or for loading data to be served by a [module
proxy](#glos-module-proxy).

By default, `download` writes nothing to standard output. It prints progress
messages and errors to standard error.

The `-json` flag causes `download` to print a sequence of JSON objects to
standard output, describing each downloaded module (or failure), corresponding
to this Go struct:

```
type Module struct {
    Path     string // module path
    Version  string // module version
    Error    string // error loading module
    Info     string // absolute path to cached .info file
    GoMod    string // absolute path to cached .mod file
    Zip      string // absolute path to cached .zip file
    Dir      string // absolute path to cached source root directory
    Sum      string // checksum for path, version (as in go.sum)
    GoModSum string // checksum for go.mod (as in go.sum)
}
```

The `-x` flag causes `download` to print the commands `download` executes
to standard error.

<a id="go-mod-edit"></a>
### `go mod edit`

<a id="go-mod-init"></a>
### `go mod init`

Usage:

```
go mod init [module-path]
```

Example:

```
go mod init
go mod init example.com/m
```

The `go mod init` command initializes and writes a new `go.mod` file in the
current directory, in effect creating a new module rooted at the current
directory. The `go.mod` file must not already exist.

`init` accepts one optional argument, the [module path](#glos-module-path) for
the new module. See [Module paths](#module-path) for instructions on choosing
a module path. If the module path argument is omitted, `init` will attempt
to infer the module path using import comments in `.go` files, vendoring tool
configuration files, and the current directory (if in `GOPATH`).

If a configuration file for a vendoring tool is present, `init` will attempt to
import module requirements from it. `init` supports the following configuration
files.

* `GLOCKFILE` (Glock)
* `Godeps/Godeps.json` (Godeps)
* `Gopkg.lock` (dep)
* `dependencies.tsv` (godeps)
* `glide.lock` (glide)
* `vendor.conf` (trash)
* `vendor.yml` (govend)
* `vendor/manifest` (gvt)
* `vendor/vendor.json` (govendor)

Vendoring tool configuration files can't always be translated with perfect
fidelity. For example, if multiple packages within the same repository are
imported at different versions, and the repository only contains one module, the
imported `go.mod` can only require the module at one version. You may wish to
run [`go list -m all`](#go-list-m) to check all versions in the [build
list](#glos-build-list), and [`go mod tidy`](#go-mod-tidy) to add missing
requirements and to drop unused requirements.

<a id="go-mod-tidy"></a>
### `go mod tidy`

<a id="go-mod-verify"></a>
### `go mod verify`

<a id="go-clean-modcache"></a>
### `go clean -modcache`

<a id="module-queries"></a>
### Module queries

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

<!-- TODO(katiehockman): why only fall back for 410/404? Either add the details
here, or write a blog post about how to build multiple types of proxies. e.g.
a "privacy preserving one" and an "authorization one" -->

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

<a id="module-cache"></a>
### Module cache

<a id="authenticating"></a>
## Authenticating modules

<!-- TODO: continue this section -->
When deciding whether to trust the source code for a module version just
fetched from a proxy or origin server, the `go` command first consults the
`go.sum` lines in the `go.sum` file of the current module. If the `go.sum` file
does not contain an entry for that module version, then it may consult the
checksum database.

<a id="go.sum-file-format"></a>
### go.sum file format

<a id="checksum-database"></a>
### Checksum database

The checksum database is a global source of `go.sum` lines. The `go` command can
use this in many situations to detect misbehavior by proxies or origin servers.

The checksum database allows for global consistency and reliability for all
publicly available module versions. It makes untrusted proxies possible since
they can't serve the wrong code without it going unnoticed. It also ensures
that the bits associated with a specific version do not change from one day to
the next, even if the module's author subsequently alters the tags in their
repository.

The checksum database is served by [sum.golang.org](https://sum.golang.org),
which is run by Google. It is a [Transparent
Log](https://research.swtch.com/tlog) (or “Merkle Tree”) of `go.sum` line
hashes, which is backed by [Trillian](https://github.com/google/trillian). The
main advantage of a Merkle tree is that independent auditors can verify that it
hasn't been tampered with, so it is more trustworthy than a simple database.

The `go` command interacts with the checksum database using the protocol
originally outlined in [Proposal: Secure the Public Go Module
Ecosystem](https://go.googlesource.com/proposal/+/master/design/25530-sumdb.md#checksum-database).

The table below specifies queries that the checksum database must respond to.
For each path, `$base` is the path portion of the checksum database URL,
`$module` is a module path, and `$version` is a version. For example, if the
checksum database URL is `https://sum.golang.org`, and the client is requesting
the record for the module `golang.org/x/text` at version `v0.3.2`, the client
would send a `GET` request for
`https://sum.golang.org/lookup/golang.org/x/text@v0.3.2`.

To avoid ambiguity when serving from case-insensitive file systems,
the `$module` and `$version` elements are
[case-encoded](https://pkg.go.dev/golang.org/x/mod/module#EscapePath)
by replacing every uppercase letter with an exclamation mark followed by the
corresponding lower-case letter. This allows modules `example.com/M` and
`example.com/m` to both be stored on disk, since the former is encoded as
`example.com/!m`.

Parts of the path surrounded by square brakets, like `[.p/$W]` denote optional
values.

<table>
  <thead>
    <tr>
      <th>Path</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>$base/latest</code></td>
      <td>
        Returns a signed, encoded tree description for the latest log. This
        signed description is in the form of a
        <a href="https://pkg.go.dev/golang.org/x/mod/sumdb/note">note</a>,
        which is text that has been signed by one or more server keys and can
        be verified using the server's public key. The tree description
        provides the size of the tree and the hash of the tree head at that
        size. This encoding is described in
        <code><a href="https://pkg.go.dev/golang.org/x/mod/sumdb/tlog#FormatTree">
        golang.org/x/mod/sumdb/tlog#FormatTree</a></code>.
      </td>
    </tr>
    <tr>
    <tr>
      <td><code>$base/lookup/$module@$version</code></td>
      <td>
        Returns the log record number for the entry about <code>$module</code>
        at <code>$version</code>, followed by the data for the record (that is,
        the <code>go.sum</code> lines for <code>$module</code> at
        <code>$version</code>) and a signed, encoded tree description that
        contains the record.
      </td>
    </tr>
    <tr>
    <tr>
      <td><code>$base/tile/$H/$L/$K[.p/$W]</code></td>
      <td>
        Returns a [log tile](https://research.swtch.com/tlog#serving_tiles),
        which is a set of hashes that make up a section of the log. Each tile
        is defined in a two-dimensional coordinate at tile level
        <code>$L</code>, <code>$K</code>th from the left, with a tile height of
        <code>$H</code>. The optional <code>.p/$W</code> suffix indicates a
        partial log tile with only <code>$W</code> hashes. Clients must fall
        back to fetching the full tile if a partial tile is not found.
      </td>
    </tr>
    <tr>
    <tr>
      <td><code>$base/tile/$H/data/$K[.p/$W]</code></td>
      <td>
        Returns the record data for the leaf hashes in
        <code>/tile/$H/0/$K[.p/$W]</code> (with a literal <code>data</code> path
        element).
      </td>
    </tr>
    <tr>
  </tbody>
</table>

If the `go` command consults the checksum database, then the first
step is to retrieve the record data through the `/lookup` endpoint. If the
module version is not yet recorded in the log, the checksum database will try
to fetch it from the origin server before replying. This `/lookup` data
provides the sum for this module version as well as its position in the log,
which informs the client of which tiles should be fetched to perform proofs.
The `go` command performs “inclusion” proofs (that a specific record exists in
the log) and “consistency” proofs (that the tree hasn’t been tampered with)
before adding new `go.sum` lines to the main module’s `go.sum` file. It's
important that the data from `/lookup` should never be used without first
authenticating it against the signed tree hash and authenticating the signed
tree hash against the client's timeline of signed tree hashes.

Signed tree hashes and new tiles served by the checksum database are stored
in the module cache, so the `go` command only needs to fetch tiles that are
missing.

The `go` command doesn't need to directly connect to the checksum database. It
can request module sums via a module proxy that
[mirrors the checksum database](https://go.googlesource.com/proposal/+/master/design/25530-sumdb.md#proxying-a-checksum-database)
and supports the protocol above. This can be particularly helpful for private,
corporate proxies which block requests outside the organization.

The `GOSUMDB` environment variable identifies the name of checksum database to use
and optionally its public key and URL, as in:

```
GOSUMDB="sum.golang.org"
GOSUMDB="sum.golang.org+<publickey>"
GOSUMDB="sum.golang.org+<publickey> https://sum.golang.org"
```

The `go` command knows the public key of `sum.golang.org`, and also that the
name `sum.golang.google.cn` (available inside mainland China) connects to the
`sum.golang.org` checksum database; use of any other database requires giving
the public key explicitly. The URL defaults to `https://` followed by the
database name.

`GOSUMDB` defaults to `sum.golang.org`, the Go checksum database run by Google.
See https://sum.golang.org/privacy for the service's privacy policy.

If `GOSUMDB` is set to `off`, or if `go get` is invoked with the `-insecure`
flag, the checksum database is not consulted, and all unrecognized modules are
accepted, at the cost of giving up the security guarantee of verified
repeatable downloads for all modules. A better way to bypass the checksum
database for specific modules is to use the `GOPRIVATE` or `GONOSUMDB`
environment variables. See [Private Modules](#private-modules) for details.

The `go env -w` command can be used to
[set these variables](/pkg/cmd/go/#hdr-Print_Go_environment_information)
for future `go` command invocations.

<a id="privacy"></a>
## Privacy

<a id="environment-variables"></a>
## Environment variables

<a id="glossary"></a>
## Glossary

<a id="glos-build-list"></a>
**build list:** The list of module versions that will be used for a build
command such as `go build`, `go list`, or `go test`. The build list is
determined from the [main module's](#glos-main-module) [`go.mod`
file](#glos-go.mod-file) and `go.mod` files in transitively required modules
using [minimal version selection](#glos-minimal-version-selection). The build
list contains versions for all modules in the [module
graph](#glos-module-graph), not just those relevant to a specific command.

<a id="glos-canonical-version"></a>
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

<a id="glos-module-cache"></a>
**module cache:** A local directory storing downloaded modules, located in
`GOPATH/pkg/mod`. See [Module cache](#module-cache).

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
