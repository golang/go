# gopls user documentation

Most of this document is written from the perspective of VSCode as at the time of writing it was the most popular editor. Most of the features described work in any editor and the settings should be easy to translate to the specifics of each editor integration.
For instance, anything in the configuration section has a specific layout, but the exact place you define the settings will depend on the editor, and the syntax of the declaration may be in a different language.

## Editors

The following is the list of editors with known integrations.
If you know of an editor not in this list that works, please let us know.

* [VSCode](vscode.md)
* [Vim / Neovim](vim.md)
* [Emacs](emacs.md)
* [Acme](acme.md)
* [Sublime Text](subl.md)

## Installing

For the most part you should not need to install or update gopls, your editor should handle that step for you.

If you do want to get the latest stable version of gopls, change to any directory that is not in either your GOPATH or a module (a temp directory is fine), and use

```sh
go get golang.org/x/tools/gopls@latest
```

**do not** use the `-u` flag, as it will update your dependencies to incompatible versions.

If you see this error:

```sh
$ go get golang.org/x/tools/gopls@latest
go: cannot use path@version syntax in GOPATH mode
```
then run
```sh
GO111MODULE=on go get golang.org/x/tools gopls@latest
```


## Configuring

gopls can be configured in a few different ways:
* environment variables

  These are often inherited from the editor that launches gopls, and sometimes the editor has a way to add or replace values before launching.

  gopls does not use the environment directly, but it can use `go list` extensively underneath, so the standard Go environment is important.

* command line

  See the [command line](#command-line) section for more information about the flags you might specify.
  All editors support some way of adding flags to gopls, for the most part you should not need to do this unless you have very unusual requirements or are trying to [troubleshoot](troubleshooting.md#steps) gopls behavior.

* editor settings

  For the most part these will be things that control how the editor interacts with or uses the results of gopls, not things that modify gopls itself. This means they are not standardized across editors, and you will have to look at the specific instructions for your editor integration to change them.

* the set of workspace folders

  This is one of the most important pieces of configuration. It is the set of folders that gopls considers to be "roots" that it should consider files to be a part of.

  If you are using modules there should be one of these per go.mod that you are working on.
  If you do not open the right folders, very little will work. **This is the most common mis-configuration of gopls that we see**.

* global configuration

  There should be a way of declaring global settings for gopls inside the editor.
  The settings block will be called "gopls" and contains a collection of controls for gopls that the editor is not expected to understand or control.

  In VSCode this would be a section in your settings file that might look like

  ```json5
  "gopls": {
    "usePlaceholders": true, // add parameter placeholders when completing a function
    "wantCompletionDocumentation": true // for documentation in completion items
  },
  ```

  See the [settings](#settings) for more information about what values you can set here.

* per workspace folder configuration

  This contains exactly the same set of values that are in the global configuration, but it is fetched for every workspace folder separately.
  The editor can choose to respond with different values per folder, but this is


### Settings

**buildFlags** *array of strings*

This is the set of flags passed on to the build system when invoked.
It is applied to things like `go list` queries when discovering files.
The most common use is to set `-tags`.

**env** *map of string to value*

This can be used to add environment variables. These will not affect gopls itself, but will be used for any external commands it invokes.

**experimentalDisabledAnalyses** *map*

The keys in this map indicate analysis passes that should be disabled.
You can use this to turn off analyses that you feel are not useful in the editor.
The values of the map are ignored.

**hoverKind** *string*

This controls the information that appears in the hover text.
It must be one of:
* "NoDocumentation"
* "SingleLine"
* "SynopsisDocumentation"
* "FullDocumentation"
* "Structured"

**useDeepCompletions** *boolean*

If true this turns on the ability to return completions from deep inside relevant entities, rather than just the locally accessible ones, for instance it may suggest fields of local variables that match.

**usePlaceholders** *boolean*

If true then completion responses may contain placeholders inside their snippets.

**wantCompletionDocumentation** *boolean*

If true it indicates that the user wants documentation with their completion responses.

**wantSuggestedFixes** *boolean*

If true this turns on the ability for the analysis system to suggest fixes rather than just report problems.
If supported by the editor, theses fixes can be automatically applied or applied with a single action.

**wantUnimportedCompletions** *boolean*

If true the completion engine is allowed to make suggestions for packages that you do not currently import.

## Command line

gopls supports much of its functionality on the command line as well.

It does this for two main reasons, firstly so that you do not have to reach for another tool to do something gopls can already do in your editor.
It also makes it easy to reproduce behavior seen in the editor from a command line you can ask others to run.

It is not a goal of gopls to be a high performance command line tool, its command line it intended for single file/package user interaction speeds, not bulk processing.

<!--- TODO: command line
detailed command line instructions, use cases and flags
--->
