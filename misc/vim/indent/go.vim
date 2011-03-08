" Copyright 2011 The Go Authors. All rights reserved.
" Use of this source code is governed by a BSD-style
" license that can be found in the LICENSE file.
"
" indent/go.vim: Vim indent file for Go.
"

if exists("b:did_indent")
    finish
endif
let b:did_indent = 1

" C indentation is mostly correct
setlocal cindent

" Options set:
" +0 -- Don't indent continuation lines (because Go doesn't use semicolons
"       much)
" L0 -- Don't move jump labels (NOTE: this isn't correct when working with
"       gofmt, but it does keep struct literals properly indented.)
" :0 -- Align case labels with switch statement
" l1 -- Always align case body relative to case labels
" J1 -- Indent JSON-style objects (properly indents struct-literals)
" (0, Ws -- Indent lines inside of unclosed parentheses by one shiftwidth
" m1 -- Align closing parenthesis line with first non-blank of matching
"       parenthesis line
"
" Known issue: Trying to do a multi-line struct literal in a short variable
"              declaration will not indent properly.
setlocal cinoptions+=+0,L0,:0,l1,J1,(0,Ws,m1
