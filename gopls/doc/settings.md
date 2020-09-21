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

<!-- BEGIN User: DO NOT MANUALLY EDIT THIS SECTION -->
### **buildFlags** *[]string*
buildFlags is the set of flags passed on to the build system when invoked.
It is applied to queries like `go list`, which is used when discovering files.
The most common use is to set `-tags`.


Default: `[]`.
### **env** *[]string*
env adds environment variables to external commands run by `gopls`, most notably `go list`.


Default: `[]`.
### **hoverKind** *enum*
hoverKind controls the information that appears in the hover text.
SingleLine and Structured are intended for use only by authors of editor plugins.
Must be one of:

 * `"FullDocumentation"`
 * `"NoDocumentation"`
 * `"SingleLine"`
 * `"Structured"`
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
<!-- END User: DO NOT MANUALLY EDIT THIS SECTION -->

## Experimental

The below settings are considered experimental. They may be deprecated or changed in the future. They are typically used to test experimental opt-in features or to disable features.

<!-- BEGIN Experimental: DO NOT MANUALLY EDIT THIS SECTION -->
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
### **codelens** *map[string]bool*
overrides the enabled/disabled state of various code lenses. Currently, we
support several code lenses:

* `generate`: run `go generate` as specified by a `//go:generate` directive.
* `upgrade_dependency`: upgrade a dependency listed in a `go.mod` file.
* `test`: run `go test -run` for a test func.
* `gc_details`: Show the gc compiler's choices for inline analysis and escaping.

Example Usage:
```json5
"gopls": {
...
  "codelens": {
    "generate": false,  // Don't run `go generate`.
    "gc_details": true  // Show a code lens toggling the display of gc's choices.
  }
...
}
```


Default: `{"gc_details":false,"generate":true,"regenerate_cgo":true,"tidy":true,"upgrade_dependency":true,"vendor":true}`.
### **completionDocumentation** *bool*
completionDocumentation enables documentation with completion results.


Default: `true`.
### **completeUnimported** *bool*
completeUnimported enables completion for packages that you do not currently import.


Default: `true`.
### **deepCompletion** *bool*
deepCompletion If true, this turns on the ability to return completions from deep inside relevant entities, rather than just the locally accessible ones.

Consider this example:

```go
package main

import "fmt"

type wrapString struct {
    str string
}

func main() {
    x := wrapString{"hello world"}
    fmt.Printf(<>)
}
```

At the location of the `<>` in this program, deep completion would suggest the result `x.str`.


Default: `true`.
### **matcher** *enum*
matcher sets the algorithm that is used when calculating completion candidates.
Must be one of:

 * `"CaseInsensitive"`
 * `"CaseSensitive"`
 * `"Fuzzy"`


Default: `"Fuzzy"`.
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
### **symbolMatcher** *enum*
symbolMatcher sets the algorithm that is used when finding workspace symbols.
Must be one of:

 * `"CaseInsensitive"`
 * `"CaseSensitive"`
 * `"Fuzzy"`


Default: `"Fuzzy"`.
### **symbolStyle** *enum*
symbolStyle specifies what style of symbols to return in symbol requests.
Must be one of:

 * `"Dynamic"`
 * `"Full"`
 * `"Package"`


Default: `"Package"`.
### **linksInHover** *bool*
linksInHover toggles the presence of links to documentation in hover.


Default: `true`.
### **tempModfile** *bool*
tempModfile controls the use of the -modfile flag in Go 1.14.


Default: `true`.
### **importShortcut** *enum*
importShortcut specifies whether import statements should link to
documentation or go to definitions.
Must be one of:

 * `"Both"`
 * `"Definition"`
 * `"Link"`


Default: `"Both"`.
### **verboseWorkDoneProgress** *bool*
verboseWorkDoneProgress controls whether the LSP server should send
progress reports for all work done outside the scope of an RPC.


Default: `false`.
### **expandWorkspaceToModule** *bool*
expandWorkspaceToModule instructs `gopls` to expand the scope of the workspace to include the
modules containing the workspace folders. Set this to false to avoid loading
your entire module. This is particularly useful for those working in a monorepo.


Default: `true`.
### **experimentalWorkspaceModule** *bool*
experimentalWorkspaceModule opts a user into the experimental support
for multi-module workspaces.


Default: `false`.
<!-- END Experimental: DO NOT MANUALLY EDIT THIS SECTION -->
