" Copyright 2011 The Go Authors. All rights reserved.
" Use of this source code is governed by a BSD-style
" license that can be found in the LICENSE file.
"
" godoc.vim: Vim command to see godoc.

if exists("g:loaded_godoc")
  finish
endif
let g:loaded_godoc = 1

let s:buf_nr = -1
let s:last_word = ''

function! s:GodocView()
  if !bufexists(s:buf_nr)
    leftabove new
    file `="[Godoc]"`
    let s:buf_nr = bufnr('%')
  elseif bufwinnr(s:buf_nr) == -1
    leftabove split
    execute s:buf_nr . 'buffer'
    delete _
  elseif bufwinnr(s:buf_nr) != bufwinnr('%')
    execute bufwinnr(s:buf_nr) . 'wincmd w'
  endif

  setlocal filetype=godoc
  setlocal bufhidden=delete
  setlocal buftype=nofile
  setlocal noswapfile
  setlocal nobuflisted
  setlocal modifiable
  setlocal nocursorline
  setlocal nocursorcolumn
  setlocal iskeyword+=:
  setlocal iskeyword-=-

  nnoremap <buffer> <silent> K :Godoc<cr>

  au BufHidden <buffer> call let <SID>buf_nr = -1
endfunction

function! s:GodocWord(word)
  let word = a:word
  silent! let content = system('godoc ' . word)
  if v:shell_error || !len(content)
    if len(s:last_word)
      silent! let content = system('godoc ' . s:last_word.'/'.word)
      if v:shell_error || !len(content)
        echo 'No documentation found for "' . word . '".'
        return
      endif
      let word = s:last_word.'/'.word
    else
      echo 'No documentation found for "' . word . '".'
      return
    endif
  endif
  let s:last_word = word
  silent! call s:GodocView()
  setlocal modifiable
  silent! %d _
  silent! put! =content
  silent! normal gg
  setlocal nomodifiable
  setfiletype godoc
endfunction

function! s:Godoc(...)
  let word = join(a:000, ' ')
  if !len(word)
    let oldiskeyword = &iskeyword
    setlocal iskeyword+=.
    let word = expand('<cword>')
    let &iskeyword = oldiskeyword
  endif
  let word = substitute(word, '[^a-zA-Z0-9\\/._~-]', '', 'g')
  let words = split(word, '\.')
  if !len(words)
    return
  endif
  call s:GodocWord(words[0])
  if len(words) > 1
    if search('^\%(const\|var\|type\|\s\+\) ' . words[1] . '\s\+=\s')
      return
    endif
    if search('^func ' . words[1] . '(')
      return
    endif
    echo 'No documentation found for "' . word . '".'
  endif
endfunction

command! -nargs=* -range -complete=customlist,go#complete#Package Godoc :call s:Godoc(<q-args>)
nnoremap <silent> <Plug>(godoc-keyword) :<C-u>call <SID>Godoc('')<CR>

" vim:ts=4:sw=4:et
