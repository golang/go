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

### **buildFlags** *array of strings*

This is the set of flags passed on to the build system when invoked. It is applied to queries like `go list`, which is used when discovering files. The most common use is to set `-tags`.

### **env** *map of string to value*

This can be used to add environment variables. These will not affect `gopls` itself, but will be used for any external commands it invokes.

### **hoverKind** *string*

This controls the information that appears in the hover text.
It must be one of:
* `"NoDocumentation"`
* `"SynopsisDocumentation"`
* `"FullDocumentation"`

Authors of editor clients may wish to handle hover text differently, and so might use different settings. The options below are not intended for use by anyone other than the authors of editor plugins.

* `"SingleLine"`
* `"Structured"`

Default: `"SynopsisDocumentation"`.

### **usePlaceholders** *boolean*

If true, then completion responses may contain placeholders for function parameters or struct fields.

Default: `false`.

### **linkTarget** *string*

This controls where points documentation for given package in `textDocument/documentLink`.
It might be one of:

* `"godoc.org"`
* `"pkg.go.dev"`
If company chooses to use its own `godoc.org`, its address can be used as well.

Default: `"pkg.go.dev"`.

### **local** *string*

This is the equivalent of the `goimports -local` flag, which puts imports beginning with this string after 3rd-party packages.
It should be the prefix of the import path whose imports should be grouped separately.

Default: `""`.

### **expandWorkspaceToModule** *boolean*

This is true if `gopls` should expand the scope of the workspace to include the
modules containing the workspace folders. Set this to false to avoid loading
your entire module. This is particularly useful for those working in a monorepo.

Default: `true`.

## Experimental

The below settings are considered experimental. They may be deprecated or changed in the future. They are typically used to test experimental opt-in features or to disable features.

### **analyses** *map[string]bool*

Analyses specify analyses that the user would like to enable or disable.
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

### **codelens** *map[string]bool*

Overrides the enabled/disabled state of various code lenses. Currently, we
support several code lenses:

* `generate`: [default: enabled] run `go generate` as specified by a `//go:generate` directive.
* `upgrade_dependency`: [default: enabled] upgrade a dependency listed in a `go.mod` file.
* `test`: [default: disabled] run `go test -run` for a test func.
* `gc_details`: [default: disabled] Show the gc compiler's choices for inline analysis and escaping.

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
### **completionDocumentation** *boolean*

If false, indicates that the user does not want documentation with completion results.

Default value: `true`.

### **completeUnimported** *boolean*

If true, the completion engine is allowed to make suggestions for packages that you do not currently import.

Default: `false`.

### **deepCompletion** *boolean*

If true, this turns on the ability to return completions from deep inside relevant entities, rather than just the locally accessible ones.

Default: `true`.

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

### **fuzzyMatching** *boolean*

If true, this enables server side fuzzy matching of completion candidates.

Default: `true`.

### **matcher** *string*

Defines the algorithm that is used when calculating completion candidates. Must be one of:

* `"fuzzy"`
* `"caseSensitive"`
* `"caseInsensitive"`

Default: `"caseInsensitive"`

### **annotations** *map[string]bool*

**noBounds** suppresses gc_details diagnostics about bounds checking.

**noEscape** suppresses gc_details diagnostics about escape analysis.

**noInline** suppresses gc_details diagnostics about inlining.

**noNilcheck** suppresses gc_details diagnostics about generated nil checks.

### **staticcheck** *boolean*

If true, it enables the use of the staticcheck.io analyzers.

### **symbolMatcher** *string*

Defines the algorithm that is used when calculating workspace symbol results. Must be one of:

* `"fuzzy"`
* `"caseSensitive"`
* `"caseInsensitive"`

Default: `"caseInsensitive"`.
