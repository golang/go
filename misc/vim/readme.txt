Vim syntax highlighting for Go (http://golang.org)
==================================================

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
  ln -s $GOROOT/misc/vim/ftdetect/gofiletype.vim $HOME/.vim/ftdetect/
  ln -s $GOROOT/misc/vim/syntax/go.vim $HOME/.vim/syntax
  echo "syntax on" >> $HOME/.vimrc


Vim filetype plugins for Go
===========================

To install one of the available filetype plugins for Go:

  1. Same as 1 above.
  2. Copy or link one or more plugins from ftplugin/go/*.vim to the
     Go-specific ftplugin directory underneath your vim runtime directory
     (normally $HOME/.vim/ftplugin/go/*.vim).
  3. Add the following line to your .vimrc file (normally $HOME/.vimrc):

     filetype plugin on


Vim indentation plugin for Go
=============================

To install automatic indentation for Go:

  1. Same as 1 above.
  2. Copy or link indent/go.vim to the indent directory underneath your vim
     runtime directory (normally $HOME/.vim/indent).
  3. Add the following line to your .vimrc file (normally $HOME/.vimrc):

     filetype indent on
