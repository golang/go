/*

Command goimports updates your Go import lines,
adding missing ones and removing unreferenced ones.

     $ go get golang.org/x/tools/cmd/goimports

It's a drop-in replacement for your editor's gofmt-on-save hook.
It has the same command-line interface as gofmt and formats
your code in the same way.

For emacs, make sure you have the latest go-mode.el:
   https://github.com/dominikh/go-mode.el
Then in your .emacs file:
   (setq gofmt-command "goimports")
   (add-to-list 'load-path "/home/you/somewhere/emacs/")
   (require 'go-mode-load)
   (add-hook 'before-save-hook 'gofmt-before-save)

For vim, set "gofmt_command" to "goimports":
    https://golang.org/change/39c724dd7f252
    https://golang.org/wiki/IDEsAndTextEditorPlugins
    etc

For GoSublime, follow the steps described here:
    http://michaelwhatcott.com/gosublime-goimports/

For other editors, you probably know what to do.

To exclude directories in your $GOPATH from being scanned for Go
files, goimports respects a configuration file at
$GOPATH/src/.goimportsignore which may contain blank lines, comment
lines (beginning with '#'), or lines naming a directory relative to
the configuration file to ignore when scanning. No globbing or regex
patterns are allowed. Use the "-v" verbose flag to verify it's
working and see what goimports is doing.

File bugs or feature requests at:

    https://golang.org/issues/new?title=x/tools/cmd/goimports:+

Happy hacking!

*/
package main // import "golang.org/x/tools/cmd/goimports"
