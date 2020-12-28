# Features

This document describes some of the features supported by `gopls`. It is
currently under construction, so, for a comprehensive list, see the
[Language Server Protocol](https://microsoft.github.io/language-server-protocol/).

For now, only special features outside of the LSP are described below.

## Special features

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

<!--TODO(rstambler): Automatically generate a list of supported features.-->
