# gopls design documentation

## Goals

* `gopls` should **become the default editor backend** for the major editors used by Go programmers, fully supported by the Go team.
* `gopls` will be a **full implementation of LSP**, as described in the [LSP specification], to standardize as many of its features as possible.
* `gopls` will be **clean and extensible** so that it can encompass additional features in the future, allowing Go tooling to become best in class once more.
* `gopls` will **support alternate build systems and file layouts**, allowing Go development to be simpler and more powerful in any environment.


## Context

While Go has a number of excellent and useful command-line tools that enhance the developer experience, it has become clear that integrating these tools with IDEs can pose challenges.

Support of these tools has relied on the goodwill of community members, and they have been put under a large burden of support at times as the language, toolchain and environments change. As a result many tools have ceased to work, have had support problems, or become confusing with forks and replacements, or provided an experience that is not as good as it could be.
See the section below on [existing solutions](#existing-solutions) for more problems and details.

This is fine for tools used occasionally, but for core IDE features, this is not acceptable.
Autocompletion, jump to definition, formatting, and other such features should always work, as they are key for Go development.

The Go team will create an editor backend that works in any build system.
It will also be able to improve upon the latency of Go tools, since each tool will no longer have to individually run the type-checker on each invocation, instead there will be a long-running process and data can be shared between the definitions, completions, diagnostics, and other features.

By taking ownership of these tools and packaging them together in the form of gopls, the Go team will ensure that the Go development experience isn’t unnecessarily complicated for Go users.
Having one editor backend will simplify the lives of Go developers, the Go team, and the maintainers of Go editor plugins.

See Rebecca's excellent GopherCon keynote [talk] and [slides] for some more context.

## Non-Goals

* Command line speed

  Although gopls will have a command line mode, it will be optimized for long running and not command responsiveness, as such it may not be the right tool for things like CI systems.
  For such cases there will have to be an alternate tool using the same underlying libraries for consistency.

* Low memory environments

  In order to do a good job of processing large projects with very low latencies gopls will be holding a lot of information in memory.
  It is presumed that developers are normally working on systems with significant RAM and this will not be a problem.
  In general this is upheld by the large memory usage of existing IDE solutions (like IntelliJ)

* Syntax highlighting

  At the moment there is no editor that delegates this functionality to a separate binary, and no standard way of doing it.

## Existing solutions

Every year the Go team conducts a survey, asking developers about their experiences with the language.

One question that is asked is “How do you feel about your editor?”.

The responses told a very negative story. Some categorized quotes:

* Setup
  * "Hard to install and configure"
  * "Inadequate documentation"
* Performance
  * "Performance is very poor"
  * "Pretty slow in large projects"
* Reliability
  * "Features work one day, but not the next"
  * "Tooling is not updated with new language features"

Each editor has its own plugin that shells out to a variety of tools, many of which break with new Go releases or because they are no longer maintained.

The individual tools each have to do the work to understand the code and all its transitive dependencies.

Each feature is a different tool, with a different set of patterns for its command line, a different way to accept input and parse output, a different way of specifying source code locations.
To support its existing feature set, VSCode installed 24 different command line tools, many of which have options or forks to configure. When looking at the set of tools that needed to be migrated to modules, across all the editors, there were 63 separate tools.

All these tools need to understand the code, and they use the same standard libraries to do it. Those libraries are optimized for these kinds of tools, but even so processing that much code takes a lot of time time. Almost none of the tools are capable of returning results within 100ms.
As developers type in their editor, multiple of these features need to activate, which means they are not just paying the cost once, but many times. The overall effect is an editing experience that feels sluggish, and features that are either not enabled or sometimes produce results that appear so slowly they are no longer useful when they arrive. This is a problem that increases with the size of the code base, which means it is getting worse over time, and is especially bad for the kinds of large code bases companies are dealing with as they use Go for more major tasks.

## Requirements

### Complete feature set

For gopls to be considered a success it has to implement the full feature set discussed [below](#Features).
This is the set of features that users need in order to feel as productive as they were with the tooling it is replacing. It does not include every feature of previous implementations, there are some features that are almost never used that should be dropped (like guru's pointer analysis) and some other features that do not easily fit and will have to be worked around (replacing the save hook/linter).

### Equivalent or better experience

For all of those features, the user experience must match or exceed the current one available in all editors.
This is an easy statement to make, but a hard one to validate or measure. Many of the possible measures fail to capture the experience.

For instance, if an attempt was made to measure the latency of a jump to definition call, the results would be fairly consistent from the old godef tool. From the gopls implementation there may be a much larger range of latencies, with the best being orders of magnitude faster, and the worse slightly worse, because gopls attempts to do far more work, but manages to cache it across calls.

Or for a completion call, it might be slower but produce a better first match such that users accept it more often, resulting in an overall better experience.

For the most part this has to rely on user reports. If users are refusing to switch because the experience is not better, it is clearly not done, if they are switching but most people are complaining, there are probably enough areas that are better to make the switch compelling but other areas which are worse. If most people are switching and either staying silent or being positive, it is probably done. When writing tools, the user is all that matters.

### Solid community of contributors

The scope and scale of the problem gopls is trying to solve is untenable for the core Go team, it is going to require a strong community to make it all happen.

This implies the code must be easy to contribute to, and easy for many developers to work on in parallel. The functionality needs to be well decoupled, and have a thorough testing story.

### Latencies that fall within user tolerance

There has been a lot of research on acceptable latencies for user actions.
<!-- TODO: research links -->
The main result that affects gopls is that feedback in direct response to continuous user actions needs to be under 100ms to be imperceptible, and anything above 200ms aggravates the user.
This means in general the aim has to be <100ms for anything that happens as the developer types.
There will always be cases where gopls fails to meet this deadline, and there needs to be ways to make the user experience okay in those cases, but in general the point of this deadline is to inform the basic architecture design, any solution that cannot theoretically meet this goal in the long term is the wrong answer.

### Easy to configure

Developers are very particular, and have very differing desires in their coding experience. gopls is going to have to support a significant amount of flexibility, in order to meet those desires.
The default settings however with no configuration at all must be the one that is best experience for most users, and where possible the features must be flexible without configuration so that the client can easily make the choices about treatment without changing its communication with gopls.

## Difficulties

### Volume of data

<!-- TODO: project sizes -->
* Small:
* Medium:
* Large:
* Corporate mono-repo: Much much bigger

Parsing and type checking large amounts of code is quite expensive, and the converted forms use a lot of space. As gopls has to keep updating this information while the developer types, it needs to manage how it caches the converted forms very carefully to balance memory use vs speed.

### Cache invalidation

The basic unit of operation for the type checking is the package, but the basic unit of operation for an editor is the file.
gopls needs to be able to map files to packages efficiently, so that when files change it knows which packages need to be updated (along with any other packages that transitively depended on them).
This is made especially difficult by the fact that changing the content of a file can modify which packages it is considered part of (either by changing the package declaration or the build tags), a file can be in more than one package, and changes can be made to files without using the editor, in which case it will not notify us of the changes.

### Inappropriate core functionality

The base libraries for Go (things like [go/token], [go/ast] and [go/types]) are all designed for compiler-like applications.
They tend to worry more about throughput than memory use, they have structures that are intended to grow and then be thrown away at program exit, and they are not designed to keep going in the presence of errors in the source they are handling.
They also have no abilities to do incremental changes.

Making a long running service work well with those libraries is a very large challenge, but writing new libraries would be far more work, and cause a significant long term cost as both sets of libraries would have to be maintained. Right now it is more important to get a working tool into the hands of users. In the long term this decision may have to be revisited, new low level libraries may be the only way to keep pushing the capabilities forwards.

### Build system capabilities

gopls is supposed to be build system agnostic, but it must use the build system to discover how files map to packages. When it tries to do so, even when the functionality is the same, the costs (in time, CPU and memory) are very different, and can significantly impact the user experience. Designing how gopls interacts with the build system to try to minimize or hide these differences is hard.

### Build tags

The build tag system in Go is quite powerful, and has many use cases. Source files can exclude themselves using powerful boolean logic on the set of active tags.
It is however designed for specifying the set of active tags on the command line, and the libraries are all designed to cope with only one valid combination at a time. There is also no way to work out the set of valid combinations.

Type checking a file requires knowledge of all the other files in the same package, and that set of files is modified by the build tags. The set of exported identifiers of a package is also affected by which files are in the package, and thus its build tags.

This means that even for files or packages that have no build tag controls it is not possible to produce correct results without knowing the set of build tags to consider.
This makes it very hard to produce useful results when viewing a file.

### Features not supported by LSP

There are some things it would be good to be able to do that do not fit easily into the existing LSP protocol.
For instance, displaying control flow information, automatic struct tags, complex refactoring...

Each feature will have to be considered carefully, and either propose a change to LSP, or add a way to have gopls specific extensions to the protocol that are still easy to use in all the editor plugins.

To avoid these at the start, only core LSP features will be implemented, as they are sufficient to meet the baseline requirements anyway, but the potential features need to be kept in mind in the core architecture.

### Distribution

Making sure that users are using the right version of gopls is going to be a problem. Each editor plugin is probably going to install the tools in its own way, some will choose to install it system wide, some will keep their own copy.

Because it is a brand new tool, it will be changing rapidly. If users are not informed they are on an old version they will be experiencing problems that have already been fixed, which is worse for them, and then probably reporting them, which wastes time for the gopls team. There needs to be a mechanism for gopls to check if is up to date, and a recommended way to install an up to date version.

### Debugging user problems

gopls is essentially a very stateful long running server on the developer's machine. Its basic operation is affected by many things, from the users environment to the contents of the local build cache. The data it is operating on is often a confidential code base that cannot be shared.
All of these things make it hard for users to report a bug usefully, or create a minimal reproduction.

There needs to be easy ways for users to report what information they can, and ways to attempt to reproduce problems without their entire state. This is also needed to produce regression tests.


## Basic design decisions

There are some fundamental architecture decisions that affect much of the rest of the design of the tool, making fundamental trade offs that impact the user experience.

### Process lifetime: *managed by the editor*

Processing a large code base to fully type check and then analyze it within the latency requirements is not feasible, and is one of the primary problems with the existing solutions. This remains true even if the computed information was cached on disk, as running analyzers and type checkers ends up requiring the full AST of all files in the dependency graph.
It is theoretically possible to do better, but only with a major re-write of the existing parsing and type checking libraries, something that is not feasible at this time.

This implies that gopls should be a long running process, that is able to cache and pre-calculate results in memory so that when a request arrives it can produce the answer much faster.

It could run as a daemon on the user's machine, but there are a lot of issues with managing a daemon. It may well be the right choice in the long term, and it should be allowed for in the fundamental architecture design, but to start with it will instead have a process that lasts as long as the editor that starts it, and that can easily be restarted.

### Caching: *in memory*

Persistent disk caches are very expensive to maintain, and require solving a lot of extra problems.
Although building the information required is expensive compared to the latencies required of the requests, it is fairly minor compared to the startup times of an editor, so it is expected that rebuilding the information when gopls is restarted will be acceptable.

The advantage gained from this is that gopls becomes stateless across restarts which means if it has issues or gets its state confused, a simple restart will often fix the problem.
It also means that when users report problems, the entire state of the on disk cache is not needed to diagnose and reproduce the issue.

### Communication: *stdin/stdout JSON*

The LSP specification defines the JSON messages that are normally used, but it does not define how those message should be sent, and there are implementations of the LSP that do not use JSON (for instance, Protocol buffers are an option).

The constraints on gopls are that it must be easy to integrate into *every editor* on *all operating systems*, and that it should not have large external dependencies.

JSON is part of the Go standard library, and is also the native language of LSP, so it makes the most sense. By far the best supported communication mechanism is the standard input and output of a process, and the common client implementations all have ways of using [JSON rpc 2] in this mode.  There were no complete and low dependency implementations of this protocol in Go, but it is a fairly small protocol on top of the JSON library that can be implemented with a moderate effort, and would be a generally useful library to have anyway.

In the future it is expected to run in separated client server mode, so writing it in a way that could use sockets instead of stdin/stdout from the start was the best way to make sure it remained possible. It was also a huge debugging aid to be able to run the gopls server by hand and watch/debug it outside the editor.

### Running other tools: *no*

<!--- TODO: subprocess discuss --->

## Features

There is a set of features that gopls needs to expose to be a comprehensive IDE solution.
The following is the minimum set of features, along with their existing solutions and how they should map to the LSP.

### Introspection

Introspection features tell developers information about their code while they work. They do not make or suggest changes.

---
Diagnostics | Static analysis results of the code, including compilation and lint errors
----------- | ---
Requires    | Full go/analysis run, which needs full AST, type and SSA information
LSP         | [`textDocument/publishDiagnostics`]
Previous    | `go build`, `go vet`, `golint`, [errcheck], [staticcheck] <!-- TODO: and all the rest -->
|           | This is one of the most important IDE features, allowing fast turn around without having to run compilers and checkers in the shell. Often used to power problem lists, gutter markers and squiggle underlines in the IDE. <br/> There is some complicated design work to do in order to let users customize the set of checks being run, preferably without having to recompile the main LSP binary.

---
Hover    | Information about the code under the cursor.
-------- | ---
Requires | AST and type information for the file and all dependencies
LSP      | [`textDocument/hover`]
Previous | [godoc], [gogetdoc]
|        | Used when reading code to display information known to the compiler but not always obvious from the code. For instance it may return the types of identifiers, or the documentation.

---
Signature help | Function parameter information and documentation
-------------- | ---
Requires       | AST and type information for the file and all dependencies
LSP            | [`textDocument/signatureHelp`]
Previous       | [gogetdoc]
|              | As a function call is being typed into code, it is helpful to know the parameters of that call to enable the developer to call it correctly.


### Navigation

Navigation features are designed to make it easier for a developer to find their way round a code base.

---
Definition | Select an identifier, and jump to the code where that identifier was defined.
---------- | ---
Requires   | Full type information for file and all dependencies
LSP        | [`textDocument/declaration`]
|          | [`textDocument/definition`]
|          | [`textDocument/typeDefinition`]
Previous   | [godef] |
|          | Asking the editor to open the place where a symbol was defined is one of the most commonly used code navigation tools inside an IDE when available. It is especially valuable when exploring an unfamiliar code base.<br/>Due to a limitation of the compiler output, it is not possible to use the binary data for this task (specifically it does not know column information) and thus it must parse from source.

---
Implementation | Reports the types that implement an interface
-------------- | ---
Requires       | Full workspace type knowledge
LSP            | [`textDocument/implementation`]
Previous       | [impl]
|              | This feature is hard to scale up to large code bases, and is going to take thought to get right. It may be feasible to implemented a more limited form in the meantime.

---
Document symbols | Provides the set of top level symbols in the current file.
---------------- | ---
Requires         | AST of the current file only
LSP              | [`textDocument/documentSymbol`]
Previous         | [go-outline], [go-symbols]
|                | Used to drive things like outline mode.

---
References | Find all references to the symbol under the cursor.
---------- | ---
Requires   | AST and type information for the **reverse** transitive closure
LSP        | [`textDocument/references`]
Previous   | [guru]
|          | This requires knowledge of every package that could possible depend on any packages the current file is part of. In the past this has been implemented either by global knowledge, which does not scale, or by specifying a "scope" which confused users to the point where they just did not use the tools. gopls is probably going to need a more powerful solution in the long term, but to start with automatically limiting the scope may produce acceptable results. This would probably be the module if known, or some sensible parent directory otherwise.


---
Folding  | Report logical hierarchies of blocks
-------- | ---
Requires | AST of the current file only
LSP      | [`textDocument/foldingRange`]
Previous | [go-outline]
|        | This is normally used to provide expand and collapse behavior in editors.

---
Selection | Report regions of logical selection around the cursor
--------- | ---
Requires  | AST of the current file only
LSP       | [`textDocument/selectionRange`]
Previous  | [guru]
|         | Used in editor features like expand selection.


### Edit assistance

These features suggest or apply edits to the code for the user, including refactoring features, for which there are many potential use cases.
Refactoring is one of the places where Go tools could potentially be very strong, but have not been so far, and thus there is huge potential for improvements in the developer experience.
There is not yet a clear understanding of the kinds of refactoring people need or how they should express them however, and there are weaknesses in the LSP protocol around this.
This means it may be much more of a research project.


---
Format   | Fix the formatting of the file
-------- | ---
Requires | AST of current file
LSP      | [`textDocument/formatting`]
|        | [`textDocument/rangeFormatting`]
|        | [`textDocument/onTypeFormatting`]
Previous | [gofmt], [goimports], [goreturns]
|        | It will use the standard format package. <br/> Current limitations are that it does not work on malformed code. It may need some very careful changes to the formatter to allow for formatting an invalid AST or changes to force the AST to a valid mode. These changes would improve range and file mode as well, but are basically vital to onTypeFormatting


---
Imports  | Rewrite the imports block automatically to match the symbols used.
-------- | ---
Requires | AST of the current file and full symbol knowledge for all candidate packages.
LSP      | [`textDocument/codeAction`]
Previous | [goimports], [goreturns]
|        | This needs knowledge of packages that are not yet in use, and the ability to find those packages by name. <br/> It also needs exported symbol information for all the packages it discovers. <br/> It should be implemented using the standard imports package, but there may need to be exposed a more fine grained API than just a file rewrite for some of the interactions.


---
Autocompletion | Makes suggestions to complete the entity currently being typed.
-------------- | ---
Requires       | AST and type information for the file and all dependencies<br/> Also full exported symbol knowledge for all packages.
LSP            | [`textDocument/completion`]
|              | [`completionItem/resolve`]
Previous       | [gocode]
|              | Autocomplete is one of the most complicated features, and the more it knows the better its suggestions can be. For instance it can autocomplete into packages that are not yet being imported if it has their public symbols. It can make better suggestions of options if it knows what kind of program you are writing. It can suggest better arguments if it knows how you normally call a function. It can suggest entire patterns of code if it knows they are common. Unlike many other features, which have a specific task, and once it is doing that task the feature is done, autocomplete will never be finished. Balancing and improving both the candidates and how they are ranked will be a research problem for a long time to come.

---
Rename   | Rename an identifier
-------- | ---
Requires | AST and type information for the **reverse** transitive closure
LSP      | [`textDocument/rename`]
|        | [`textDocument/prepareRename`]
Previous | [gorename]
|        | This uses the same information that find references does, with all the same problems and limitations. It is slightly worse because the changes it suggests make it intolerant of incorrect results. It is also dangerous using it to change the public API of a package.

---
Suggested fixes | Suggestions that can be manually or automatically accepted to change the code
--------------- | ---
Requires        | Full go/analysis run, which needs full AST, type and SSA information
LSP             | [`textDocument/codeAction`]
Previous        | N/A
|               | This is a brand new feature powered by the new go/analysis engine, and it should allow a huge amount of automated refactoring.


[LSP specification]: https://microsoft.github.io/language-server-protocol/specifications/specification-3-14/
[talk]: TODO
[slides]: https://github.com/gophercon/2019-talks/blob/master/RebeccaStambler-GoPleaseStopBreakingMyEditor/slides.pdf "Go, please stop breaking my editor!"
[JSON rpc 2]: https://www.jsonrpc.org/specification

[errcheck]: https://github.com/kisielk/errcheck
[go-outline]: https://github.com/lukehoban/go-outline
[go-symbols]: https://github.com/acroca/go-symbols
[gocode]: https://github.com/stamblerre/gocode
[godef]: https://github.com/rogpeppe/godef
[godoc]: https://golang.org/cmd/godoc
[gofmt]: https://golang.org/cmd/gofmt
[gogetdoc]: https://github.com/zmb3/gogetdoc
[goimports]: https://godoc.org/golang.org/x/tools/cmd/goimports
[gorename]: https://godoc.org/golang.org/x/tools/cmd/gorename
[goreturns]: https://github.com/sqs/goreturns
[gotags]: https://github.com/jstemmer/gotags
[guru]: https://godoc.org/golang.org/x/tools/cmd/guru
[impl]: https://github.com/josharian/impl
[staticcheck]: https://staticcheck.io/docs/
[go/types]: https://golang.org/pkg/go/types/
[go/ast]: https://golang.org/pkg/go/ast/
[go/token]: https://golang.org/pkg/go/token/


[`completionItem/resolve`]:https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#completionItem_resolve
[`textDocument/codeAction`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_codeAction
[`textDocument/completion`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_completion
[`textDocument/declaration`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_declaration
[`textDocument/definition`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_definition
[`textDocument/documentLink`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_documentLink
[`textDocument/documentSymbol`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_documentSymbol
[`textDocument/foldingRange`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_foldingRange
[`textDocument/formatting`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_formatting
[`textDocument/highlight`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_highlight
[`textDocument/hover`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_hover
[`textDocument/implementation`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_implementation
[`textDocument/onTypeFormatting`]:https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_onTypeFormatting
[`textDocument/prepareRename`]:https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_prepareRename
[`textDocument/publishDiagnostics`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_publishDiagnostics
[`textDocument/rangeFormatting`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_rangeFormatting
[`textDocument/references`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_references
[`textDocument/rename`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_rename
[`textDocument/selectionRange`]:https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_selectionRange
[`textDocument/signatureHelp`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_signatureHelp
[`textDocument/typeDefinition`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_typeDefinition
[`workspace/didChangeWatchedFiles`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#workspace_didChangeWatchedFiles
