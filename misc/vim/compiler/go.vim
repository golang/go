" Copyright 2013 The Go Authors. All rights reserved.
" Use of this source code is governed by a BSD-style
" license that can be found in the LICENSE file.
"
" compiler/go.vim: Vim compiler file for Go.

if exists("current_compiler")
    finish
endif
let current_compiler = "go"

if exists(":CompilerSet") != 2
    command -nargs=* CompilerSet setlocal <args>
endif

let s:save_cpo = &cpo
set cpo-=C

CompilerSet makeprg=go\ build
CompilerSet errorformat=
        \%-G#\ %.%#,
        \%A%f:%l:%c:\ %m,
        \%A%f:%l:\ %m,
        \%C%*\\s%m,
        \%-G%.%#

let &cpo = s:save_cpo
unlet s:save_cpo

" vim:ts=4:sw=4:et
