Vim plugins for Go (http://golang.org)
======================================

To use all the Vim plugins, add these lines to your $HOME/.vimrc.

  " Some Linux distributions set filetype in /etc/vimrc.
  " Clear filetype flags before changing runtimepath to force Vim to reload them.
  filetype off
  filetype plugin indent off
  set runtimepath+=$GOROOT/misc/vim
  filetype plugin indent on
  syntax on

If you want to select fewer plugins, use the instructions in the rest of
this file.

Vim syntax highlighting
-----------------------

To install automatic syntax highlighting for GO programs:

  1. Copy or link the filetype detection script to the ftdetect directory
     underneath your vim runtime directory (normally $HOME/.vim/ftdetect)
  2. Copy or link syntax/go.vim to the syntax directory underneath your vim
     runtime directory (normally $HOME/.vim/syntax). Linking this file rather
     than just copying it will ensure any changes are automatically reflected
     in your syntax highlighting.
  3. Add the following line to your .vimrc file (normally $HOME/.vimrc):

     syntax on

In a typical unix environment you might accomplish this using the following
commands:

  mkdir -p $HOME/.vim/ftdetect
  mkdir -p $HOME/.vim/syntax
  mkdir -p $HOME/.vim/autoload/go
  ln -s $GOROOT/misc/vim/ftdetect/gofiletype.vim $HOME/.vim/ftdetect/
  ln -s $GOROOT/misc/vim/syntax/go.vim $HOME/.vim/syntax
  ln -s $GOROOT/misc/vim/autoload/go/complete.vim $HOME/.vim/autoload/go
  echo "syntax on" >> $HOME/.vimrc


Vim filetype plugins
--------------------

To install one of the available filetype plugins:

  1. Same as 1 above.
  2. Copy or link one or more plugins from ftplugin/go/*.vim to the
     Go-specific ftplugin directory underneath your vim runtime directory
     (normally $HOME/.vim/ftplugin/go/*.vim).
  3. Add the following line to your .vimrc file (normally $HOME/.vimrc):

     filetype plugin on


Vim indentation plugin
----------------------

To install automatic indentation:

  1. Same as 1 above.
  2. Copy or link indent/go.vim to the indent directory underneath your vim
     runtime directory (normally $HOME/.vim/indent).
  3. Add the following line to your .vimrc file (normally $HOME/.vimrc):

     filetype indent on


Godoc plugin
------------

To install godoc plugin:

  1. Same as 1 above.
  2. Copy or link plugin/godoc.vim to $HOME/.vim/plugin/godoc,
     syntax/godoc.vim to $HOME/.vim/syntax/godoc.vim,
     and autoload/go/complete.vim to $HOME/.vim/autoload/go/complete.vim.
