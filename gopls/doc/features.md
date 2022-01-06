# Features

This document describes some of the features supported by `gopls`. It is
currently under construction, so, for a comprehensive list, see the
[Language Server Protocol](https://microsoft.github.io/language-server-protocol/).

## Special features

Here, only special features outside of the LSP are described.

### Symbol Queries

Gopls supports some extended syntax for `workspace/symbol` requests, when using
the `fuzzy` symbol matcher (the default). Inspired by the popular fuzzy matcher
[FZF](https://github.com/junegunn/fzf), the following special characters are
supported within symbol queries:

| Character | Usage     | Match        |
| --------- | --------- | ------------ |
| `'`       | `'abc`    | exact        |
| `^`       | `^printf` | exact prefix |
| `$`       | `printf$` | exact suffix |

## Template Files

Gopls provides some support for Go template files, that is, files that
are parsed by `text/template` or `html/template`.
Gopls recognizes template files based on their file extension, which may be
configured by the
[`templateExtensions`](https://github.com/golang/tools/blob/master/gopls/doc/settings.md#templateextensions-string) setting.
Making this list empty turns off template support.

In template files, template support works inside
the default `{{` delimiters. (Go template parsing
allows the user to specify other delimiters, but
gopls does not know how to do that.)

Gopls template support includes the following features:
+ **Diagnostics**: if template parsing returns an error,
it is presented as a diagnostic. (Missing functions do not produce errors.)
+ **Syntax Highlighting**: syntax highlighting is provided for template files.
+  **Definitions**: gopls provides jump-to-definition inside templates, though it does not understand scoping (all templates are considered to be in one global scope).
+  **References**: gopls provides find-references, with the same scoping limitation as definitions.
+ **Completions**: gopls will attempt to suggest completions inside templates.

### Configuring your editor

In addition to configuring `templateExtensions`, you may need to configure your
editor or LSP client to activate `gopls` for template files. For example, in
`VS Code` you will need to configure both
[`files.associations`](https://code.visualstudio.com/docs/languages/identifiers)
and `build.templateExtensions` (the gopls setting).

<!--TODO(rstambler): Automatically generate a list of supported features.-->

