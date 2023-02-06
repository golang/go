# Semantic Tokens

The [LSP](https://microsoft.github.io/language-server-protocol/specifications/specification-3-17/#textDocument_semanticTokens)
specifies semantic tokens as a way of telling clients about language-specific
properties of pieces of code in a file being edited.

The client asks for a set of semantic tokens and modifiers. This note describe which ones
gopls will return, and under what circumstances. Gopls has no control over how the client
converts semantic tokens into colors (or some other visible indication). In vscode it
is possible to modify the color a theme uses by setting the `editor.semanticTokenColorCustomizations`
object. We provide a little [guidance](#Colors) later.

There are 22 semantic tokens, with 10 possible modifiers. The protocol allows each semantic
token to be used with any of the 1024 subsets of possible modifiers, but most combinations
don't make intuitive sense (although `async documentation` has a certain appeal).

The 22 semantic tokens are `namespace`, `type`, `class`, `enum`, `interface`,
		`struct`, `typeParameter`, `parameter`, `variable`, `property`, `enumMember`,
		`event`, `function`, `method`, `macro`, `keyword`, `modifier`, `comment`,
		`string`, `number`, `regexp`, `operator`.

The 10 modifiers are `declaration`, `definition`, `readonly`, `static`,
		`deprecated`, `abstract`, `async`, `modification`, `documentation`, `defaultLibrary`.

The authoritative lists are in the [specification](https://microsoft.github.io/language-server-protocol/specifications/specification-3-17/#semanticTokenTypes)

For the implementation to work correctly the client and server have to agree on the ordering
of the tokens and of the modifiers. Gopls, therefore, will only send tokens and modifiers
that the client has asked for. This document says what gopls would send if the client
asked for everything. By default, vscode asks for everything.

Gopls sends 11 token types for `.go` files and 1 for `.*tmpl` files.
Nothing is sent for any other kind of file.
This all could change. (When Go has generics, gopls will return `typeParameter`.)

For `.*tmpl` files gopls sends `macro`, and no modifiers, for each `{{`...`}}` scope.

## Semantic tokens for Go files

There are two contrasting guiding principles that might be used to decide what to mark
with semantic tokens. All clients already do some kind of syntax marking. E.g., vscode
uses a TextMate grammar. The minimal principle would send semantic tokens only for those
language features that cannot be reliably found without parsing Go and looking at types.
The maximal principle would attempt to convey as much as possible about the Go code,
using all available parsing and type information.

There is much to be said for returning minimal information, but the minimal principle is
not well-specified. Gopls has no way of knowing what the clients know about the Go program
being edited. Even in vscode the TextMate grammars can be more or less elaborate
and change over time. (Nonetheless, a minimal implementation would not return `keyword`,
`number`, `comment`, or `string`.)

The maximal position isn't particularly well-specified either. To chose one example, a
format string might have formatting codes (`%[4]-3.6f`), escape sequences (`\U00010604`), and regular
characters. Should these all be distinguished? One could even imagine distinguishing
different runes by their Unicode language assignment, or some other Unicode property, such as
being [confusable](http://www.unicode.org/Public/security/10.0.0/confusables.txt).

Gopls does not come close to either of these principles.  Semantic tokens are returned for
identifiers, keywords, operators, comments, and literals. (Semantic tokens do not
cover the file. They are not returned for
white space or punctuation, and there is no semantic token for labels.)
The following describes more precisely what gopls
does, with a few notes on possible alternative choices.
The references to *object* refer to the
```types.Object``` returned by the type checker. The references to *nodes* refer to the
```ast.Node``` from the parser.

1. __`keyword`__ All Go [keywords](https://golang.org/ref/spec#Keywords) are marked `keyword`.
1. __`namespace`__ All package names are marked `namespace`. In an import, if there is an
alias, it would be marked. Otherwise the last component of the import path is marked.
1. __`type`__ Objects of type ```types.TypeName``` are marked `type`.
If they are also ```types.Basic```
the modifier is `defaultLibrary`. (And in ```type B struct{C}```, ```B``` has modifier `definition`.)
1. __`parameter`__ The formal arguments in ```ast.FuncDecl``` and ```ast.FuncType``` nodes are marked `parameter`.
1. __`variable`__  Identifiers in the
scope of ```const``` are modified with `readonly`. ```nil``` is usually a `variable` modified with both
`readonly` and `defaultLibrary`. (```nil``` is a predefined identifier; the user can redefine it,
in which case it would just be a variable, or whatever.) Identifiers of type ```types.Variable``` are,
not surprisingly, marked `variable`. Identifiers being defined (node ```ast.GenDecl```) are modified
by `definition` and, if appropriate, `readonly`. Receivers (in method declarations) are
`variable`.
1. __`method`__ Methods are marked at their definition (```func (x foo) bar() {}```) or declaration
in an ```interface```. Methods are not marked where they are used.
In ```x.bar()```, ```x``` will be marked
either as a `namespace` if it is a package name, or as a `variable` if it is an interface value,
so distinguishing ```bar``` seemed superfluous.
1. __`function`__ Bultins (```types.Builtin```) are modified with `defaultLibrary`
(e.g., ```make```, ```len```, ```copy```). Identifiers whose
object is ```types.Func``` or whose node is ```ast.FuncDecl``` are `function`.
1. __`comment`__ Comments and struct tags. (Perhaps struct tags should be `property`?)
1. __`string`__ Strings. Could add modifiers for e.g., escapes or format codes.
1. __`number`__ Numbers. Should the ```i``` in ```23i``` be handled specially?
1. __`operator`__ Assignment operators, binary operators, ellipses (```...```), increment/decrement
operators, sends (```<-```), and unary operators.

Gopls will send the modifier `deprecated` if it finds a comment
```// deprecated``` in the godoc.

The unused tokens for Go code are `class`, `enum`, `interface`,
		`struct`, `typeParameter`, `property`, `enumMember`,
		`event`, `macro`, `modifier`,
		`regexp`

## Colors

These comments are about vscode.

The documentation has a [helpful](https://code.visualstudio.com/api/language-extensions/semantic-highlight-guide#custom-textmate-scope-mappings)
description of which semantic tokens correspond to scopes in TextMate grammars. Themes seem
to use the TextMate scopes to decide on colors.

Some examples of color customizations are [here](https://medium.com/@danromans/how-to-customize-semantic-token-colorization-with-visual-studio-code-ac3eab96141b).

## Note

While a file is being edited it may temporarily contain either
parsing errors or type errors. In this case gopls cannot determine some (or maybe any)
of the semantic tokens. To avoid weird flickering it is the responsibility
of clients to maintain the semantic token information
in the unedited part of the file, and they do.