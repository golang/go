# Settings

<!--TODO: Generate this file from the documentation in golang/org/x/tools/internal/lsp/source/options.go.-->

This document describes the global settings for `gopls` inside the editor. The settings block will be called `"gopls"` and contains a collection of controls for `gopls` that the editor is not expected to understand or control. These settings can also be configured differently per workspace folder.

In VSCode, this would be a section in your `settings.json` file that might look like this:

```json5
  "gopls": {
    "usePlaceholders": true,
    "completeUnimported": true
  },
```

## Officially supported

Below is the list of settings that are officially supported for `gopls`.

To enable all experimental features, use **allExperiments: `true`**. You will
still be able to independently override specific experimental features.

<!-- BEGIN User: DO NOT MANUALLY EDIT THIS SECTION -->
### **buildFlags** *[]string*
buildFlags is the set of flags passed on to the build system when invoked.
It is applied to queries like `go list`, which is used when discovering files.
The most common use is to set `-tags`.


Default: `[]`.
### **env** *map[string]string*
env adds environment variables to external commands run by `gopls`, most notably `go list`.


Default: `{}`.
### **hoverKind** *enum*
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
### **usePlaceholders** *bool*
placeholders enables placeholders for function parameters or struct fields in completion responses.


Default: `false`.
### **linkTarget** *string*
linkTarget controls where documentation links go.
It might be one of:

* `"godoc.org"`
* `"pkg.go.dev"`

If company chooses to use its own `godoc.org`, its address can be used as well.


Default: `"pkg.go.dev"`.
### **local** *string*
local is the equivalent of the `goimports -local` flag, which puts imports beginning with this string after 3rd-party packages.
It should be the prefix of the import path whose imports should be grouped separately.


Default: `""`.
### **gofumpt** *bool*
gofumpt indicates if we should run gofumpt formatting.


Default: `false`.
### **analyses** *map[string]bool*
analyses specify analyses that the user would like to enable or disable.
A map of the names of analysis passes that should be enabled/disabled.
A full list of analyzers that gopls uses can be found [here](analyzers.md)

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
### **codelenses** *map[string]bool*
codelenses overrides the enabled/disabled state of code lenses. See the "Code Lenses"
section of settings.md for the list of supported lenses.

Example Usage:
```json5
"gopls": {
...
  "codelens": {
    "generate": false,  // Don't show the `go generate` lens.
    "gc_details": true  // Show a code lens toggling the display of gc's choices.
  }
...
}
```


Default: `{"gc_details":false,"generate":true,"regenerate_cgo":true,"tidy":true,"upgrade_dependency":true,"vendor":true}`.
### **linksInHover** *bool*
linksInHover toggles the presence of links to documentation in hover.


Default: `true`.
### **importShortcut** *enum*
importShortcut specifies whether import statements should link to
documentation or go to definitions.
Must be one of:

 * `"Both"`
 * `"Definition"`
 * `"Link"`


Default: `"Both"`.
### **matcher** *enum*
matcher sets the algorithm that is used when calculating completion candidates.
Must be one of:

 * `"CaseInsensitive"`
 * `"CaseSensitive"`
 * `"Fuzzy"`


Default: `"Fuzzy"`.
### **symbolMatcher** *enum*
symbolMatcher sets the algorithm that is used when finding workspace symbols.
Must be one of:

 * `"CaseInsensitive"`
 * `"CaseSensitive"`
 * `"Fuzzy"`


Default: `"Fuzzy"`.
### **symbolStyle** *enum*
symbolStyle controls how symbols are qualified in symbol responses.

Example Usage:
```json5
"gopls": {
...
  "symbolStyle": "dynamic",
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
<!-- END User: DO NOT MANUALLY EDIT THIS SECTION -->

## Experimental

The below settings are considered experimental. They may be deprecated or changed in the future. They are typically used to test experimental opt-in features or to disable features.

<!-- BEGIN Experimental: DO NOT MANUALLY EDIT THIS SECTION -->
### **annotations** *map[string]bool*
annotations suppress various kinds of optimization diagnostics
that would be reported by the gc_details command.
 * noNilcheck suppresses display of nilchecks.
 * noEscape suppresses escape choices.
 * noInline suppresses inlining choices.
 * noBounds suppresses bounds checking diagnostics.


Default: `{}`.
### **staticcheck** *bool*
staticcheck enables additional analyses from staticcheck.io.


Default: `false`.
### **semanticTokens** *bool*
semanticTokens controls whether the LSP server will send
semantic tokens to the client.


Default: `false`.
### **expandWorkspaceToModule** *bool*
expandWorkspaceToModule instructs `gopls` to adjust the scope of the
workspace to find the best available module root. `gopls` first looks for
a go.mod file in any parent directory of the workspace folder, expanding
the scope to that directory if it exists. If no viable parent directory is
found, gopls will check if there is exactly one child directory containing
a go.mod file, narrowing the scope to that directory if it exists.


Default: `true`.
### **experimentalWorkspaceModule** *bool*
experimentalWorkspaceModule opts a user into the experimental support
for multi-module workspaces.


Default: `false`.
### **experimentalDiagnosticsDelay** *time.Duration*
experimentalDiagnosticsDelay controls the amount of time that gopls waits
after the most recent file modification before computing deep diagnostics.
Simple diagnostics (parsing and type-checking) are always run immediately
on recently modified packages.

This option must be set to a valid duration string, for example `"250ms"`.


Default: `"250ms"`.
### **experimentalPackageCacheKey** *bool*
experimentalPackageCacheKey controls whether to use a coarser cache key
for package type information to increase cache hits. This setting removes
the user's environment, build flags, and working directory from the cache
key, which should be a safe change as all relevant inputs into the type
checking pass are already hashed into the key. This is temporarily guarded
by an experiment because caching behavior is subtle and difficult to
comprehensively test.


Default: `true`.
### **allowModfileModifications** *bool*
allowModfileModifications disables -mod=readonly, allowing imports from
out-of-scope modules. This option will eventually be removed.


Default: `false`.
### **allowImplicitNetworkAccess** *bool*
allowImplicitNetworkAccess disables GOPROXY=off, allowing implicit module
downloads rather than requiring user action. This option will eventually
be removed.


Default: `false`.
<!-- END Experimental: DO NOT MANUALLY EDIT THIS SECTION -->

## Debugging

The below settings are for use in debugging `gopls`. Like the experimental options, they may be deprecated or changed in the future.

<!-- BEGIN Debugging: DO NOT MANUALLY EDIT THIS SECTION -->
### **verboseOutput** *bool*
verboseOutput enables additional debug logging.


Default: `false`.
### **completionBudget** *time.Duration*
completionBudget is the soft latency goal for completion requests. Most
requests finish in a couple milliseconds, but in some cases deep
completions can take much longer. As we use up our budget we
dynamically reduce the search scope to ensure we return timely
results. Zero means unlimited.


Default: `"100ms"`.
<!-- END Debugging: DO NOT MANUALLY EDIT THIS SECTION -->

## Code Lenses

These are the code lenses that `gopls` currently supports. They can be enabled and disabled using the `codeLenses` setting, documented above. The names and features are subject to change.

<!-- BEGIN Lenses: DO NOT MANUALLY EDIT THIS SECTION -->
### **Run go generate**
Identifier: `generate`

generate runs `go generate` for a given directory.


### **Regenerate cgo**
Identifier: `regenerate_cgo`

regenerate_cgo regenerates cgo definitions.


### **Run test(s)**
Identifier: `test`

test runs `go test` for a specific test function.


### **Run go mod tidy**
Identifier: `tidy`

tidy runs `go mod tidy` for a module.


### **Upgrade dependency**
Identifier: `upgrade_dependency`

upgrade_dependency upgrades a dependency.


### **Run go mod vendor**
Identifier: `vendor`

vendor runs `go mod vendor` for a module.


### **Toggle gc_details**
Identifier: `gc_details`

gc_details controls calculation of gc annotations.


<!-- END Lenses: DO NOT MANUALLY EDIT THIS SECTION -->
