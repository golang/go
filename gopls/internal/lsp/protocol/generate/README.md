# LSP Support for gopls

## The protocol

The LSP protocol exchanges json-encoded messages between the client and the server.
(gopls is the server.) The messages are either Requests, which require Responses, or
Notifications, which generate no response. Each Request or Notification has a method name
such as "textDocument/hover" that indicates its meaning and determines which function in the server will handle it.
The protocol is described in a
[web page](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.18/specification/),
in words, and in a json file (metaModel.json) available either linked towards the bottom of the
web page, or in the vscode-languageserver-node repository. This code uses the latter so the
exact version can be tied to a githash. By default, the command will download the `github.com/microsoft/vscode-languageserver-node` repository to a temporary directory.

The specification has five sections

1. Requests, which describe the Request and Response types for request methods (e.g., *textDocument/didChange*),
2. Notifications, which describe the Request types for notification methods,
3. Structures, which describe named struct-like types,
4. TypeAliases, which describe type aliases,
5. Enumerations, which describe named constants.

Requests and Notifications are tagged with a Method (e.g., `"textDocument/hover"`).
The specification does not specify the names of the functions that handle the messages. These
names are specified by the `methodNames` map. Enumerations generate Go `const`s, but
in Typescript they are scoped to namespaces, while in Go they are scoped to a package, so the Go names
may need to be modified to avoid name collisions. (See the `disambiguate` map, and its use.)

Finally, the specified types are Typescript types, which are quite different from Go types.

### Optionality

The specification can mark fields in structs as Optional. The client distinguishes between missing
fields and `null` fields in some cases. The Go translation for an optional type
should be making sure the field's value
can be `nil`, and adding the json tag `,omitempty`. The former condition would be satisfied by
adding `*` to the field's type if the type is not a reference type.

### Types

The specification uses a number of different types, only a few of which correspond directly to Go types.
The specification's types are "base", "reference", "map", "literal", "stringLiteral", "tuple", "and", "or".
The "base" types correspond directly to Go types, although some Go types needs to be chosen for `URI` and `DocumentUri`. (The "base" types`RegExp`, `BooleanLiteral`, `NumericLiteral` never occur.)

"reference" types are the struct-like types in the Structures section of the specification. The given
names are suitable for Go to use, except the code needs to change names like `_Initialze` to `XInitialize` so
they are exported for json marshaling and unmarshaling.

"map" types are just like Go. (The key type in all of them is `DocumentUri`.)

"stringLiteral" types are types whose type name and value are a single string. The chosen Go equivalent
is to make the type `string` and the value a constant. (The alternative would be to generate a new
named type, which seemed redundant.)

"literal" types are like Go anonymous structs, so they have to be given a name. (All instances
of the remaining types have to be given names. One approach is to construct the name from the components
of the type, but this leads to misleading punning, and is unstable if components are added. The other approach
is to construct the name from the context of the definition, that is, from the types it is defined within.
For instance `Lit__InitializeParams_clientInfo` is the "literal" type at the
`clientInfo` field in the `_InitializeParams`
struct. Although this choice is sensitive to the ordering of the components, the code uses this approach,
presuming that reordering components is an unlikely protocol change.)

"tuple" types are generated as Go structs. (There is only one, with two `uint32` fields.)

"and" types are Go structs with embedded type names. (There is only one, `And_Param_workspace_configuration`.)

"or" types are the most complicated. There are a lot of them and there is no simple Go equivalent.
They are defined as structs with a single `Value interface{}` field and custom json marshaling
and unmarshaling code. Users can assign anything to `Value` but the type will be checked, and
correctly marshaled, by the custom marshaling code. The unmarshaling code checks types, so `Value`
will have one of the permitted types. (`nil` is always allowed.) There are about 40 "or" types that
have a single non-null component, and these are converted to the component type.

## Processing

The code parses the json specification file, and scans all the types. It assigns names, as described
above, to the types that are unnamed in the specification, and constructs Go equivalents as required.
(Most of this code is in typenames.go.)

There are four output files. tsclient.go and tsserver.go contain the definition and implementation
of the `protocol.Client` and `protocol.Server` types and the code that dispatches on the Method
of the Request or Notification. tsjson.go contains the custom marshaling and unmarshaling code.
And tsprotocol.go contains the type and const definitions.

### Accommodating gopls

As the code generates output, mostly in generateoutput.go and main.go,
it makes adjustments so that no changes are required to the existing Go code.
(Organizing the computation this way makes the code's structure simpler, but results in
a lot of unused types.)
There are three major classes of these adjustments, and leftover special cases.

The first major
adjustment is to change generated type names to the ones gopls expects. Some of these don't change the
semantics of the type, just the name.
But for historical reasons a lot of them replace "or" types by a single
component of the type. (Until fairly recently Go only saw or used only one of components.)
The `goplsType` map in tables.go controls this process.

The second major adjustment is to the types of fields of structs, which is done using the
`renameProp` map in tables.go.

The third major adjustment handles optionality, controlling `*` and `,omitempty` placement when
the default rules don't match what gopls is expecting. (The map is `goplsStar`, also in tables.go)
(If the intermediate components in expressions of the form `A.B.C.S` were optional, the code would need
a lot of useless checking for nils. Typescript has a language construct to avoid most checks.)

Then there are some additional special cases. There are a few places with adjustments to avoid
recursive types. For instance `LSPArray` is `[]LSPAny`, but `LSPAny` is an "or" type including `LSPArray`.
The solution is to make `LSPAny` an `interface{}`. Another instance is `_InitializeParams.trace`
whose type is an "or" of 3 stringLiterals, which just becomes a `string`.

### Checking

`TestAll(t *testing.T)` checks that there are no unexpected fields in the json specification.

While the code is executing, it checks that all the entries in the maps in tables.go are used.
It also checks that the entries in `renameProp` and `goplsStar` are not redundant.

As a one-time check on the first release of this code, diff-ing the existing and generated tsclient.go
and tsserver.go code results in only whitespace and comment diffs. The existing and generated
tsprotocol.go differ in whitespace and comments, and in a substantial number of new type definitions
that the older, more heuristic, code did not generate. (And the unused type `_InitializeParams` differs
slightly between the new and the old, and is not worth fixing.)

### Some history

The original stub code was written by hand, but with the protocol under active development, that
couldn't last. The web page existed before the json specification, but it lagged the implementation
and was hard to process by machine. So the earlier version of the generating code was written in Typescript, and
used the Typescript compiler's API to parse the protocol code in the repository.
It then used a set of heuristics
to pick out the elements of the protocol, and another set of overlapping heuristics to create the Go code.
The output was functional, but idiosyncratic, and the code was fragile and barely maintainable.

### The future

Most of the adjustments using the maps in tables.go could be removed by making changes, mostly to names,
in the gopls code. Using more "or" types in gopls requires more elaborate, but stereotyped, changes.
But even without all the adjustments, making this its own module would face problems; a number of
dependencies would have to be factored out. And, it is fragile. The custom unmarshaling code knows what
types it expects. A design that return an 'any' on unexpected types would match the json
'ignore unexpected values' philosophy better, but the Go code would need extra checking.
