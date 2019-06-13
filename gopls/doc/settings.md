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

## **usePlaceholders** *boolean*

If true, then completion responses may contain placeholders for function parameters or struct fields.

Default: `false`.

## Experimental

The below settings are considered experimental. They may be deprecated or changed in the future. They are typically used to test experimental opt-in features or to disable features.

### **experimentalDisabledAnalyses** *array of strings*

A list of the names of analysis passes that should be disabled. You can use this to turn off analyses that you feel are not useful in the editor.

### **staticcheck** *boolean*

If true, it enables the use of the staticcheck.io analyzers.

### **completionDocumentation** *boolean*

If false, indicates that the user does not want documentation with completion results.

Default value: `true`.

**completeUnimported** *boolean*

If true, the completion engine is allowed to make suggestions for packages that you do not currently import.

Default: `false`.

### **deepCompletion** *boolean*

If true, this turns on the ability to return completions from deep inside relevant entities, rather than just the locally accessible ones. Consider this example:

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
