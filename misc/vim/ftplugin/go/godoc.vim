" Copyright 2011 The Go Authors. All rights reserved.
" Use of this source code is governed by a BSD-style
" license that can be found in the LICENSE file.
"
" godoc.vim: Vim command to see godoc.

if exists("b:did_ftplugin")
    finish
endif

silent! nmap <buffer> <silent> K <Plug>(godoc-keyword)

" vim:ts=4:sw=4:et
