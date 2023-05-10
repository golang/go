# Settings

<!--TODO: Generate this file from the documentation in golang.org/x/tools/gopls/internal/lsp/source/options.go.-->

This document describes the global settings for `gopls` inside the editor.
The settings block will be called `"gopls"` and contains a collection of
controls for `gopls` that the editor is not expected to understand or control.
These settings can also be configured differently per workspace folder.

In VSCode, this would be a section in your `settings.json` file that might look
like this:

```json5
  "gopls": {
    "ui.completion.usePlaceholders": true,
     ...
  },
```

## Officially supported

Below is the list of settings that are officially supported for `gopls`.

Any settings that are experimental or for debugging purposes are marked as
such.

To enable all experimental features, use **allExperiments: `true`**. You will
still be able to independently override specific experimental features.

<!-- BEGIN User: DO NOT MANUALLY EDIT THIS SECTION -->

* [Build](#build)
* [Formatting](#formatting)
* [UI](#ui)
  * [Completion](#completion)
  * [Diagnostic](#diagnostic)
  * [Documentation](#documentation)
  * [Inlayhint](#inlayhint)
  * [Navigation](#navigation)

### Build

#### **buildFlags** *[]string*

buildFlags is the set of flags passed on to the build system when invoked.
It is applied to queries like `go list`, which is used when discovering files.
The most common use is to set `-tags`.

Default: `[]`.

#### **env** *map[string]string*

env adds environment variables to external commands run by `gopls`, most notably `go list`.

Default: `{}`.

#### **directoryFilters** *[]string*

directoryFilters can be used to exclude unwanted directories from the
workspace. By default, all directories are included. Filters are an
operator, `+` to include and `-` to exclude, followed by a path prefix
relative to the workspace folder. They are evaluated in order, and
the last filter that applies to a path controls whether it is included.
The path prefix can be empty, so an initial `-` excludes everything.

DirectoryFilters also supports the `**` operator to match 0 or more directories.

Examples:

Exclude node_modules at current depth: `-node_modules`

Exclude node_modules at any depth: `-**/node_modules`

Include only project_a: `-` (exclude everything), `+project_a`

Include only project_a, but not node_modules inside it: `-`, `+project_a`, `-project_a/node_modules`

Default: `["-**/node_modules"]`.

#### **templateExtensions** *[]string*

templateExtensions gives the extensions of file names that are treateed
as template files. (The extension
is the part of the file name after the final dot.)

Default: `[]`.

#### **memoryMode** *enum*

**This setting is experimental and may be deleted.**

memoryMode controls the tradeoff `gopls` makes between memory usage and
correctness.

Values other than `Normal` are untested and may break in surprising ways.

Must be one of:

* `"DegradeClosed"`: In DegradeClosed mode, `gopls` will collect less information about
packages without open files. As a result, features like Find
References and Rename will miss results in such packages.
* `"Normal"`

Default: `"Normal"`.

#### **expandWorkspaceToModule** *bool*

**This setting is experimental and may be deleted.**

expandWorkspaceToModule instructs `gopls` to adjust the scope of the
workspace to find the best available module root. `gopls` first looks for
a go.mod file in any parent directory of the workspace folder, expanding
the scope to that directory if it exists. If no viable parent directory is
found, gopls will check if there is exactly one child directory containing
a go.mod file, narrowing the scope to that directory if it exists.

Default: `true`.

#### **allowModfileModifications** *bool*

**This setting is experimental and may be deleted.**

allowModfileModifications disables -mod=readonly, allowing imports from
out-of-scope modules. This option will eventually be removed.

Default: `false`.

#### **allowImplicitNetworkAccess** *bool*

**This setting is experimental and may be deleted.**

allowImplicitNetworkAccess disables GOPROXY=off, allowing implicit module
downloads rather than requiring user action. This option will eventually
be removed.

Default: `false`.

#### **standaloneTags** *[]string*

standaloneTags specifies a set of build constraints that identify
individual Go source files that make up the entire main package of an
executable.

A common example of standalone main files is the convention of using the
directive `//go:build ignore` to denote files that are not intended to be
included in any package, for example because they are invoked directly by
the developer using `go run`.

Gopls considers a file to be a standalone main file if and only if it has
package name "main" and has a build directive of the exact form
"//go:build tag" or "// +build tag", where tag is among the list of tags
configured by this setting. Notably, if the build constraint is more
complicated than a simple tag (such as the composite constraint
`//go:build tag && go1.18`), the file is not considered to be a standalone
main file.

This setting is only supported when gopls is built with Go 1.16 or later.

Default: `["ignore"]`.

### Formatting

#### **local** *string*

local is the equivalent of the `goimports -local` flag, which puts
imports beginning with this string after third-party packages. It should
be the prefix of the import path whose imports should be grouped
separately.

Default: `""`.

#### **gofumpt** *bool*

gofumpt indicates if we should run gofumpt formatting.

Default: `false`.

### UI

#### **codelenses** *map[string]bool*

codelenses overrides the enabled/disabled state of code lenses. See the
"Code Lenses" section of the
[Settings page](https://github.com/golang/tools/blob/master/gopls/doc/settings.md#code-lenses)
for the list of supported lenses.

Example Usage:

```json5
"gopls": {
...
  "codelenses": {
    "generate": false,  // Don't show the `go generate` lens.
    "gc_details": true  // Show a code lens toggling the display of gc's choices.
  }
...
}
```

Default: `{"gc_details":false,"generate":true,"regenerate_cgo":true,"tidy":true,"upgrade_dependency":true,"vendor":true}`.

#### **semanticTokens** *bool*

**This setting is experimental and may be deleted.**

semanticTokens controls whether the LSP server will send
semantic tokens to the client.

Default: `false`.

#### **noSemanticString** *bool*

**This setting is experimental and may be deleted.**

noSemanticString turns off the sending of the semantic token 'string'

Default: `false`.

#### **noSemanticNumber** *bool*

**This setting is experimental and may be deleted.**

noSemanticNumber  turns off the sending of the semantic token 'number'

Default: `false`.

#### Completion

##### **usePlaceholders** *bool*

placeholders enables placeholders for function parameters or struct
fields in completion responses.

Default: `false`.

##### **completionBudget** *time.Duration*

**This setting is for debugging purposes only.**

completionBudget is the soft latency goal for completion requests. Most
requests finish in a couple milliseconds, but in some cases deep
completions can take much longer. As we use up our budget we
dynamically reduce the search scope to ensure we return timely
results. Zero means unlimited.

Default: `"100ms"`.

##### **matcher** *enum*

**This is an advanced setting and should not be configured by most `gopls` users.**

matcher sets the algorithm that is used when calculating completion
candidates.

Must be one of:

* `"CaseInsensitive"`
* `"CaseSensitive"`
* `"Fuzzy"`

Default: `"Fuzzy"`.

##### **experimentalPostfixCompletions** *bool*

**This setting is experimental and may be deleted.**

experimentalPostfixCompletions enables artificial method snippets
such as "someSlice.sort!".

Default: `true`.

#### Diagnostic

##### **analyses** *map[string]bool*

analyses specify analyses that the user would like to enable or disable.
A map of the names of analysis passes that should be enabled/disabled.
A full list of analyzers that gopls uses can be found in
[analyzers.md](https://github.com/golang/tools/blob/master/gopls/doc/analyzers.md).

Example Usage:

```json5
...
"analyses": {
  "unreachable": false, // Disable the unreachable analyzer.
  "unusedparams": true  // Enable the unusedparams analyzer.
}
...
```

Default: `{}`.

##### **staticcheck** *bool*

**This setting is experimental and may be deleted.**

staticcheck enables additional analyses from staticcheck.io.
These analyses are documented on
[Staticcheck's website](https://staticcheck.io/docs/checks/).

Default: `false`.

##### **annotations** *map[string]bool*

**This setting is experimental and may be deleted.**

annotations specifies the various kinds of optimization diagnostics
that should be reported by the gc_details command.

Can contain any of:

* `"bounds"` controls bounds checking diagnostics.
* `"escape"` controls diagnostics about escape choices.
* `"inline"` controls diagnostics about inlining choices.
* `"nil"` controls nil checks.

Default: `{"bounds":true,"escape":true,"inline":true,"nil":true}`.

##### **vulncheck** *enum*

**This setting is experimental and may be deleted.**

vulncheck enables vulnerability scanning.

Must be one of:

* `"Imports"`: In Imports mode, `gopls` will report vulnerabilities that affect packages
directly and indirectly used by the analyzed main module.
* `"Off"`: Disable vulnerability analysis.

Default: `"Off"`.

##### **diagnosticsDelay** *time.Duration*

**This is an advanced setting and should not be configured by most `gopls` users.**

diagnosticsDelay controls the amount of time that gopls waits
after the most recent file modification before computing deep diagnostics.
Simple diagnostics (parsing and type-checking) are always run immediately
on recently modified packages.

This option must be set to a valid duration string, for example `"250ms"`.

Default: `"1s"`.

#### Documentation

##### **hoverKind** *enum*

hoverKind controls the information that appears in the hover text.
SingleLine and Structured are intended for use only by authors of editor plugins.

Must be one of:

* `"FullDocumentation"`
* `"NoDocumentation"`
* `"SingleLine"`
* `"Structured"` is an experimental setting that returns a structured hover format.
This format separates the signature from the documentation, so that the client
can do more manipulation of these fields.\
This should only be used by clients that support this behavior.
* `"SynopsisDocumentation"`

Default: `"FullDocumentation"`.

##### **linkTarget** *string*

linkTarget controls where documentation links go.
It might be one of:

* `"godoc.org"`
* `"pkg.go.dev"`

If company chooses to use its own `godoc.org`, its address can be used as well.

Modules matching the GOPRIVATE environment variable will not have
documentation links in hover.

Default: `"pkg.go.dev"`.

##### **linksInHover** *bool*

linksInHover toggles the presence of links to documentation in hover.

Default: `true`.

#### Inlayhint

##### **hints** *map[string]bool*

**This setting is experimental and may be deleted.**

hints specify inlay hints that users want to see. A full list of hints
that gopls uses can be found in
[inlayHints.md](https://github.com/golang/tools/blob/master/gopls/doc/inlayHints.md).

Default: `{}`.

#### Navigation

##### **importShortcut** *enum*

importShortcut specifies whether import statements should link to
documentation or go to definitions.

Must be one of:

* `"Both"`
* `"Definition"`
* `"Link"`

Default: `"Both"`.

##### **symbolMatcher** *enum*

**This is an advanced setting and should not be configured by most `gopls` users.**

symbolMatcher sets the algorithm that is used when finding workspace symbols.

Must be one of:

* `"CaseInsensitive"`
* `"CaseSensitive"`
* `"FastFuzzy"`
* `"Fuzzy"`

Default: `"FastFuzzy"`.

##### **symbolStyle** *enum*

**This is an advanced setting and should not be configured by most `gopls` users.**

symbolStyle controls how symbols are qualified in symbol responses.

Example Usage:

```json5
"gopls": {
...
  "symbolStyle": "Dynamic",
...
}
```

Must be one of:

* `"Dynamic"` uses whichever qualifier results in the highest scoring
match for the given symbol query. Here a "qualifier" is any "/" or "."
delimited suffix of the fully qualified symbol. i.e. "to/pkg.Foo.Field" or
just "Foo.Field".
* `"Full"` is fully qualified symbols, i.e.
"path/to/pkg.Foo.Field".
* `"Package"` is package qualified symbols i.e.
"pkg.Foo.Field".

Default: `"Dynamic"`.

##### **symbolScope** *enum*

symbolScope controls which packages are searched for workspace/symbol
requests. The default value, "workspace", searches only workspace
packages. The legacy behavior, "all", causes all loaded packages to be
searched, including dependencies; this is more expensive and may return
unwanted results.

Must be one of:

* `"all"` matches symbols in any loaded package, including
dependencies.
* `"workspace"` matches symbols in workspace packages only.

Default: `"all"`.

#### **verboseOutput** *bool*

**This setting is for debugging purposes only.**

verboseOutput enables additional debug logging.

Default: `false`.

<!-- END User: DO NOT MANUALLY EDIT THIS SECTION -->

#### **newDiff** *string*

newDiff enables the new diff implementation. If this is "both", for now both
diffs will be run and statistics will be generated in a file in $TMPDIR. This
is a risky setting; help in trying it is appreciated. If it is "old" the old
implementation is used, and if it is "new", just the new implementation is
used. This setting will eventually be deleted, once gopls has fully migrated to
the new diff algorithm.

Default: 'both'.

## Code Lenses

These are the code lenses that `gopls` currently supports. They can be enabled
and disabled using the `codelenses` setting, documented above. Their names and
features are subject to change.

<!-- BEGIN Lenses: DO NOT MANUALLY EDIT THIS SECTION -->
### **Toggle gc_details**

Identifier: `gc_details`

Toggle the calculation of gc annotations.
### **Run go generate**

Identifier: `generate`

Runs `go generate` for a given directory.
### **Regenerate cgo**

Identifier: `regenerate_cgo`

Regenerates cgo definitions.
### **Run govulncheck.**

Identifier: `run_govulncheck`

Run vulnerability check (`govulncheck`).
### **Run test(s) (legacy)**

Identifier: `test`

Runs `go test` for a specific set of test or benchmark functions.
### **Run go mod tidy**

Identifier: `tidy`

Runs `go mod tidy` for a module.
### **Upgrade a dependency**

Identifier: `upgrade_dependency`

Upgrades a dependency in the go.mod file for a module.
### **Run go mod vendor**

Identifier: `vendor`

Runs `go mod vendor` for a module.
<!-- END Lenses: DO NOT MANUALLY EDIT THIS SECTION -->
