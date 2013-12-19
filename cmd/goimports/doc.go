/*

Command goimports updates your Go import lines,
adding missing ones and removing unreferenced ones.

     $ go get code.google.com/p/go.tools/cmd/goimports

It's a fork of gofmt, and will also format your code, so it can be
used as a replacement for your gofmt-on-save hook in your editor of
choice.

For emacs, make sure you have the latest (Go 1.2) go-mode.el:
   https://go.googlecode.com/hg/misc/emacs/go-mode.el

Then in your .emacs file:
   (setq gofmt-command "goimports")
   (add-to-list 'load-path "/home/you/goroot/misc/emacs/")
   (require 'go-mode-load)
   (add-hook 'before-save-hook 'gofmt-before-save)

For vim, set "gofmt_command" to "goimports":

    https://code.google.com/p/go/source/detail?r=39c724dd7f252
    https://code.google.com/p/go/source/browse#hg%2Fmisc%2Fvim
    etc

For GoSublime, follow the steps described here:
    http://mdwhatcott.wordpress.com/2013/12/15/installing-and-enabling-goimports-with-gosublime/

For other editors, you probably know what to do.

Happy hacking!

*/
package main
