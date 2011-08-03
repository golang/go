" Copyright 2011 The Go Authors. All rights reserved.
" Use of this source code is governed by a BSD-style
" license that can be found in the LICENSE file.

if exists("b:current_syntax")
  finish
endif

syn case match
syn match  godocTitle "^\([A-Z]*\)$"

command -nargs=+ HiLink hi def link <args>

HiLink godocTitle Title

delcommand HiLink

let b:current_syntax = "godoc"

" vim:ts=4 sts=2 sw=2:
